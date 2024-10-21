import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
from .utils import exist_and_not_none


def preprocess_data(
    data, 
    input_template=None, 
    input_key="input", 
    weight_key=None,
    reward_key=None,
    solution_key="solution",
    apply_chat_template=None
) -> str:
    
    if apply_chat_template:
        messages = data[input_key]
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        prompt = apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)
        messages = [{"role": "user", "content": prompt}]
    weight = data[weight_key] if weight_key else 1.0
    reward = data[reward_key] if reward_key else 0.0
    solution = str(data[solution_key])
    return messages, prompt, weight, reward, solution


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.input_template = input_template
        # self.n_samples_per_prompt = getattr(self.strategy.args, "n_samples_per_prompt", 1)
        self.infos = {"messages": [], "weights": [], "rewards": [], "solutions": []}
        self.prompts = []

        input_key = getattr(self.strategy.args, "input_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)
        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            message, prompt, weight, reward, solution = preprocess_data(
                data, 
                input_template, 
                input_key, 
                getattr(self.strategy.args, "weight_key", None),
                getattr(self.strategy.args, "reward_key", None),
                getattr(self.strategy.args, "solution_key", "solution"),
                apply_chat_template
            )
            self.prompts.append(prompt)
            self.infos["messages"].append(message)
            self.infos["weights"].append(weight)
            self.infos["rewards"].append(reward)
            self.infos["solutions"].append(solution)

        # Weighting normalization
        beta2 = getattr(self.strategy.args, "init_kl_coef2", 0)
        if beta2 > 0:
            for i, weight in enumerate(self.infos["weights"]):
                self.infos["weights"][i] = weight ** (1 / beta2)
        weight_mean = np.mean(self.infos["weights"])
        for i, weight in enumerate(self.infos["weights"]):
            self.infos["weights"][i] = weight / weight_mean

        self.strategy.print("Weights description:", pd.DataFrame(self.infos["weights"]).describe())
        self.strategy.print("An example of processed prompt:")
        self.strategy.print(self.prompts[0])
        self.strategy.print("Message:", str(self.infos["messages"][0]))
        self.strategy.print("Weight:", self.infos["weights"][0])
        self.strategy.print("Reward:", self.infos["rewards"][0])

    def __len__(self):
        length = len(self.prompts)
        return length # * self.n_samples_per_prompt
    
    def collate_fn(self, item_list):
        prompts, infos = [], {key: [] for key in self.infos.keys()}
        for prompt, info in item_list:
            prompts.append(prompt)
            for key in self.infos.keys():
                infos[key].append(info[key])
        return prompts, infos

    def __getitem__(self, idx):
        # idx /= self.n_samples_per_prompt
        return self.prompts[idx], {
            "messages": self.infos["messages"][idx],
            "weights": self.infos["weights"][idx],
            "rewards": self.infos["rewards"][idx],
            "solutions": self.infos["solutions"][idx],
        }
