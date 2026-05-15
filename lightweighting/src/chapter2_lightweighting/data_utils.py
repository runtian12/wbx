from __future__ import annotations

from pathlib import Path
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


def read_text_lines(path: str) -> Sequence[str]:
    texts = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            texts.append(line)
    if not texts:
        raise ValueError(f"文本文件为空: {path}")
    return texts


def build_text_file_causal_lm_loader(tokenizer, path: str, max_length: int = 128, batch_size: int = 2) -> DataLoader:
    return build_demo_causal_lm_loader(
        tokenizer=tokenizer,
        texts=read_text_lines(path),
        max_length=max_length,
        batch_size=batch_size,
    )
