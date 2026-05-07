from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import DataLoader


def build_demo_causal_lm_loader(tokenizer, texts: Sequence[str], max_length: int = 128, batch_size: int = 2) -> DataLoader:
    class DemoDataset(torch.utils.data.Dataset):
        def __init__(self, tk, seqs, max_len):
            self.tk = tk
            self.seqs = list(seqs)
            self.max_len = max_len

        def __len__(self):
            return len(self.seqs)

        def __getitem__(self, idx):
            enc = self.tk(
                self.seqs[idx],
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt",
            )
            item = {k: v.squeeze(0) for k, v in enc.items()}
            item["labels"] = item["input_ids"].clone()
            return item

    ds = DemoDataset(tokenizer, texts, max_length)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)
