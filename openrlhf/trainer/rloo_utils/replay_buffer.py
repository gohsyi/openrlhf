import random
from abc import ABC
from dataclasses import dataclass
from typing import List, Optional

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
    advatanges: (1)
    attention_mask: (S)
    action_mask: (A)

    "A" is the number of actions.
    """

    weights: torch.Tensor
    sequences: torch.Tensor
    action_log_probs: torch.Tensor
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
        seq, act_log_prob, att_mask, act_mask = (
            item.sequences,
            item.action_log_probs,
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
            item.attention_mask,
            item.action_mask,
        ) = (
            seq[left_pad:right_pad],
            act_log_prob[:right_pad],
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

    @torch.no_grad()
    def rloo(self, rloo_k: int = 8, beta1: float = 0.01, beta2: float = 0.0):
        rewards = [getattr(item, "advantages") for item in self]
        rewards = torch.stack(rewards).float().reshape(-1, rloo_k)
        baseline = (rewards.sum(1, keepdim=True) - rewards) / (rloo_k - 1)
        advantages = rewards - baseline
        advantages = advantages.flatten()
        rstd = (advantages - advantages.mean()).pow(2).mean().rsqrt()
        advantages = (advantages - advantages.mean()) * rstd

        for i, item in enumerate(self):
            setattr(item, "advantages", advantages[i])

        if beta2 > 0:
            weights = (rewards.exp().mean(1) / rewards.mean(1).exp()).pow(beta1 / beta2)
            normalized_weights = weights / weights.sum() * weights.size(-1)
            for i, item in enumerate(self):
                setattr(item, "weights", normalized_weights[i // rloo_k])
            return {
                "weights": wandb.Histogram(weights),
                "normalized_weights": wandb.Histogram(normalized_weights)
            }
        else:
            return {}
