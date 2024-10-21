import random
from abc import ABC
from dataclasses import dataclass
from typing import List, Optional, Dict

import torch
import torch.nn.functional as F

from openrlhf.models.utils import masked_mean

from .experience_maker import Experience

import wandb


@dataclass
class BufferItem:
    """BufferItem is an item of experience data.

    Shapes of each tensor:
    sequences: (S)
    action_log_probs: (A)
    values: (1)
    returns: (1)
    advatanges: (1)
    attention_mask: (S)
    action_mask: (A)

    "A" is the number of actions.
    """

    weights: torch.Tensor
    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]


def split_experience_batch(experience: Experience) -> List[BufferItem]:
    batch_size = experience.sequences.size(0)
    batch_kwargs = [{} for _ in range(batch_size)]
    keys = (
        "weights",
        "sequences",
        "action_log_probs",
        "values",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
    )
    for key in keys:
        value = getattr(experience, key)
        vals = torch.unbind(value)
        assert batch_size == len(vals)
        for i, v in enumerate(vals):
            batch_kwargs[i][key] = v

    for i in range(batch_size):
        batch_kwargs[i]["info"] = {}
    for k, v in experience.info.items():
        vals = torch.unbind(v)
        assert batch_size == len(vals)
        for i, vv in enumerate(vals):
            batch_kwargs[i]["info"][k] = vv.item()

    items = [BufferItem(**kwargs) for kwargs in batch_kwargs]
    return items


def zero_pad_sequences(sequences: List[torch.Tensor], side: str = "left") -> torch.Tensor:
    if not sequences[0].size():
        return torch.stack(sequences)
    assert side in ("left", "right")
    max_len = max(seq.size(0) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding))
    return torch.stack(padded_sequences, dim=0)


def make_experience_batch(items: List[BufferItem]) -> Experience:
    kwargs = {}
    keys = (
        "weights",
        "sequences",
        "action_log_probs",
        "values",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
    )
    for key in keys:
        vals = [getattr(item, key) for item in items]
        batch_data = zero_pad_sequences(vals, "left")
        kwargs[key] = batch_data

    kwargs["info"] = {}
    for key in items[0].info.keys():
        vals = torch.tensor([item.info[key] for item in items])
        kwargs["info"][key] = vals
    return Experience(**kwargs)


def remove_padding_in_sequences(items):
    for item in items:
        seq, act_log_prob, value, ret, adv, att_mask, act_mask = (
            item.sequences,
            item.action_log_probs,
            item.values,
            item.returns,
            item.advantages,
            item.attention_mask,
            item.action_mask,
        )
        right_pad = (1 - act_mask.long()).sum()
        right_pad = None if right_pad == 0 else -right_pad

        # left_pad for seq and att_mask
        left_pad = att_mask.long().argmax()
        (
            item.sequences,
            item.action_log_probs,
            item.values,
            item.returns,
            item.advantages,
            item.attention_mask,
            item.action_mask,
        ) = (
            seq[left_pad:right_pad],
            act_log_prob[:right_pad],
            value[:right_pad],
            ret[:right_pad],
            adv[:right_pad],
            att_mask[left_pad:right_pad],
            act_mask[:right_pad],
        )
    return items


class NaiveReplayBuffer(ABC):
    """Naive replay buffer class. It stores experience.

    Args:
        sample_batch_size (int): Batch size when sampling.
        limit (int, optional): Limit of number of experience samples. A number <= 0 means unlimited. Defaults to 0.
        cpu_offload (bool, optional): Whether to offload experience to cpu when sampling. Defaults to True.
    """

    def __init__(self, sample_batch_size: int, limit: int = 0, cpu_offload: bool = True) -> None:
        super().__init__()
        self.sample_batch_size = sample_batch_size
        # limit <= 0 means unlimited
        self.limit = limit
        self.cpu_offload = cpu_offload
        self.target_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.items: List[BufferItem] = []

    @torch.no_grad()
    def append(self, experience: Experience) -> None:
        if self.cpu_offload:
            experience.to_device(torch.device("cpu"))
        items = split_experience_batch(experience)
        items = remove_padding_in_sequences(items)
        self.items.extend(items)
        if self.limit > 0:
            samples_to_remove = len(self.items) - self.limit
            if samples_to_remove > 0:
                self.items = self.items[samples_to_remove:]

    def clear(self) -> None:
        self.items.clear()

    @torch.no_grad()
    def sample(self) -> Experience:
        items = random.sample(self.items, self.sample_batch_size)
        experience = make_experience_batch(items)
        if self.cpu_offload:
            experience.to_device(self.target_device)
        return experience

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> BufferItem:
        return self.items[idx]

    def collate_fn(self, batch) -> Experience:
        experience = make_experience_batch(batch)
        return experience

    def reweight(
        self, 
        strategy,
        micro_rollout_batch_size, 
        update_timesteps, 
        n_samples_per_prompt,
        beta1 = 0.01, 
        beta2 = 1.0
    ) -> Dict[str, wandb.Histogram]:
        """
        rewards     =  [1, 2, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6]                 (1, update_timesteps x n_samples_per_prompt x micro_rollout_batch_size)
        reshape     => [[1, 2], [1, 2], [3, 4], [3, 4], [5, 6], [5, 6]]     (update_timesteps x n_samples_per_prompt, micro_rollout_batch_size)
        transpose   => [[1, 1, 3, 3, 5, 5], [2, 2, 4, 4, 6, 6]]             (micro_rollout_batch_size, update_timesteps x n_samples_per_prompt)
        split       => [[1, 1], [2, 2]], [[3, 3], [4, 4]], [[5, 5], [6, 6]] (micro_rollout_batch_size, n_samples_per_prompt) x update_timesteps
        cat         => [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]
        """
        # beta1, beta2 = 1.0, 1.0
        N = len(self.items)
        assert N == micro_rollout_batch_size * update_timesteps * n_samples_per_prompt
        
        rewards = torch.as_tensor([item.info["reward"] for item in self.items])
        rewards = torch.cat(rewards.reshape(-1, micro_rollout_batch_size).transpose(0, 1).split(n_samples_per_prompt, dim=1))

        weights = []
        for rews in rewards:
            max_rew = rews.max()
            weights.append(
                torch.pow(torch.mean(torch.exp((rews - max_rew) / beta1)), beta1 / beta2) / 
                torch.pow(torch.exp(torch.mean((rews - max_rew) / beta1)), beta1 / beta2)
            )

        weights = torch.as_tensor(weights)
        weights_mean = strategy.all_reduce(weights.mean(), "mean")
        weights /= weights_mean

        for i, item in enumerate(self.items):
            update_timestep = i // (micro_rollout_batch_size * n_samples_per_prompt)
            idx = update_timestep * micro_rollout_batch_size + i % micro_rollout_batch_size
            setattr(item, "weights", weights[idx])

        return {
            "weights": wandb.Histogram(weights),
        }

    def normalize(self, attribute: str, strategy) -> None:
        assert attribute == "advantages"
        items = []
        action_masks = []
        for item in self:
            items.append(getattr(item, attribute))
            action_masks.append(item.action_mask)

        items_vector = torch.cat(items).float().flatten()
        action_masks_vector = torch.cat(action_masks).flatten()

        # for DP
        # mean
        sum_and_count = torch.tensor([items_vector.sum(), action_masks_vector.sum()], device=items_vector.device)
        all_sum, all_count = strategy.all_reduce(sum_and_count, "sum")
        mean = all_sum / all_count
        # std
        std = ((items_vector - mean).pow(2) * action_masks_vector).sum()
        all_std = strategy.all_reduce(std, "sum")
        rstd = (all_std / all_count).clamp(min=1e-8).rsqrt()

        for i, item in enumerate(self):
            setattr(item, attribute, (items[i] - mean) * rstd)
