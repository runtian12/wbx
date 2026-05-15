from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def count_nonzero_parameters(model: nn.Module) -> int:
    total = 0
    for param in model.parameters():
        total += int(torch.count_nonzero(param.detach()).item())
    return total


def parameter_sparsity(model: nn.Module) -> float:
    total = count_parameters(model)
    if total == 0:
        return 0.0
    nonzero = count_nonzero_parameters(model)
    return 1.0 - float(nonzero) / float(total)


@torch.no_grad()
def evaluate_causal_lm_loss(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int = 8) -> Dict[str, float]:
    model.eval()
    losses = []
    for step, batch in enumerate(loader):
        if step >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        losses.append(float(out.loss.item()))
    if not losses:
        return {"loss": 0.0, "num_batches": 0}
    return {
        "loss": float(sum(losses) / len(losses)),
        "num_batches": len(losses),
    }

