from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import DistillConfig


class SelectiveDistiller:
    """对应论文 2.3.3 节的选择性知识蒸馏。"""

    def __init__(
        self,
        teacher_model: nn.Module,
        reference_model: nn.Module,
        student_model: nn.Module,
        device: torch.device,
        config: DistillConfig,
    ):
        self.teacher = teacher_model.eval()
        self.reference = reference_model.eval()
        self.student = student_model.train()
        self.device = device
        self.config = config

    @staticmethod
    def _token_kl(teacher_logits: torch.Tensor, student_logits: torch.Tensor, temperature: float) -> torch.Tensor:
        t = temperature
        teacher_prob = F.softmax(teacher_logits / t, dim=-1)
        student_log_prob = F.log_softmax(student_logits / t, dim=-1)
        kl = F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=-1)
        return kl * (t ** 2)

    def train(self, train_loader: DataLoader) -> nn.Module:
        optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        global_step = 0
        for epoch in range(self.config.epochs):
            for batch in train_loader:
                global_step += 1
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch.get("labels", batch["input_ids"])
                valid_mask = labels.ne(-100)

                with torch.no_grad():
                    teacher_out = self.teacher(**batch)
                    ref_out = self.reference(**batch)
                stu_out = self.student(**batch)

                teacher_logits = teacher_out.logits[:, :-1, :]
                ref_logits = ref_out.logits[:, :-1, :]
                stu_logits = stu_out.logits[:, :-1, :]
                token_mask = valid_mask[:, 1:]

                ref_loss = self._token_kl(teacher_logits, ref_logits, self.config.temperature)
                stu_loss = self._token_kl(teacher_logits, stu_logits, self.config.temperature)
                delta = stu_loss - ref_loss
                delta = delta.masked_fill(~token_mask, float("-inf"))

                flat_delta = delta.view(-1)
                valid_delta = flat_delta[torch.isfinite(flat_delta)]
                keep_num = max(1, int(math.ceil(valid_delta.numel() * self.config.keep_ratio)))
                if valid_delta.numel() == 0:
                    continue

                threshold = torch.topk(valid_delta, k=keep_num).values.min()
                selected = (delta >= threshold) & token_mask

                masked_loss = stu_loss[selected]
                if masked_loss.numel() == 0:
                    continue
                loss = masked_loss.mean()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.config.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.config.grad_clip)
                optimizer.step()

                if global_step % self.config.log_every == 0:
                    print(
                        f"[SelectiveKD] epoch={epoch + 1} step={global_step} "
                        f"loss={loss.item():.6f} kept_tokens={selected.sum().item()}"
                    )
        return self.student
