"""
接受概率预测器训练脚本
基于SpecDec++的训练方法
"""

import os
import json
import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import Trainer, TrainingArguments
import numpy as np
from tqdm import tqdm

try:
    from .models import AcceptancePredictionHead
except ImportError:
    from models import AcceptancePredictionHead

IGNORE_INDEX = -100


@dataclass
class TrainConfig:
    """训练配置"""
    # 模型路径
    draft_model_path: str = "Qwen/Qwen2.5-0.5B"
    target_model_path: str = "Qwen/Qwen2.5-7B"

    # 数据路径
    train_data_path: str = "./data/train.json"
    eval_data_path: Optional[str] = None
    output_dir: str = "./output/acceptance_head"

    # 训练参数
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    # 模型参数
    num_res_layers: int = 1
    mixing_ratio: float = 0.15  # 混合目标模型token的比例
    weight_mismatch: float = 1.0  # 不匹配类别的权重

    # 其他
    seed: int = 42
    device: str = "cuda"
    bf16: bool = True


class AcceptanceDataset(Dataset):
    """
    接受概率预测数据集

    数据格式:
    {
        "prefix": [token_ids],  # 前缀token
        "tokens": [token_ids],  # 目标模型生成的token
        "draft": [token_ids],   # 草稿模型生成的token
        "p_acc": [float],       # 接受概率标签
    }
    """

    def __init__(self, data_path: str, mixing_ratio: float = 0.15):
        super().__init__()

        print(f"Loading data from {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.samples = data
        self.mixing_ratio = mixing_ratio
        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        prefix = torch.tensor(sample['prefix'], dtype=torch.long)
        tokens = torch.tensor(sample['tokens'], dtype=torch.long)
        draft = torch.tensor(sample['draft'], dtype=torch.long)

        # 接受标签: 1表示接受, 0表示拒绝
        # 比较草稿token和目标token是否一致
        labels = (draft == tokens).long()

        return {
            'prefix': prefix,
            'tokens': tokens,
            'draft': draft,
            'labels': labels,
        }


def collate_fn(batch):
    """数据批处理函数"""
    # 找到最大长度
    max_prefix_len = max(len(item['prefix']) for item in batch)
    max_seq_len = max(len(item['tokens']) for item in batch)

    batch_size = len(batch)

    # 初始化张量
    prefixes = torch.zeros(batch_size, max_prefix_len, dtype=torch.long)
    tokens = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    drafts = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    labels = torch.full((batch_size, max_seq_len), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_prefix_len + max_seq_len, dtype=torch.long)

    for i, item in enumerate(batch):
        prefix_len = len(item['prefix'])
        seq_len = len(item['tokens'])

        prefixes[i, :prefix_len] = item['prefix']
        tokens[i, :seq_len] = item['tokens']
        drafts[i, :seq_len] = item['draft']
        labels[i, :seq_len] = item['labels']
        attention_mask[i, :prefix_len + seq_len] = 1

    return {
        'prefixes': prefixes,
        'tokens': tokens,
        'drafts': drafts,
        'labels': labels,
        'attention_mask': attention_mask,
    }


class AcceptanceHeadTrainer:
    """
    接受概率预测器训练器
    """

    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # 设置随机种子
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # 加载模型
        self._load_models()

        # 初始化接受概率预测器
        self._init_acceptance_head()

    def _load_models(self):
        """加载草稿模型和目标模型"""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading draft model from {self.config.draft_model_path}")
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            self.config.draft_model_path,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float32,
            device_map="auto",
        )
        self.draft_model.eval()

        print(f"Loading target model from {self.config.target_model_path}")
        self.target_model = AutoModelForCausalLM.from_pretrained(
            self.config.target_model_path,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float32,
            device_map="auto",
        )
        self.target_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.draft_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 获取隐藏层维度
        self.hidden_size = self.draft_model.config.hidden_size

    def _init_acceptance_head(self):
        """初始化接受概率预测器"""
        acc_config = {
            'hidden_size': self.hidden_size,
            'num_layers': self.config.num_res_layers,
        }
        self.acceptance_head = AcceptancePredictionHead(acc_config).to(self.device)

    def generate_training_data(
            self,
            prompts: List[str],
            output_path: str,
            max_length: int = 128,
            num_samples_per_prompt: int = 5,
    ):
        """
        生成训练数据

        通过比较草稿模型和目标模型的输出来生成接受/拒绝标签
        """
        print("Generating training data...")

        all_samples = []

        for prompt in tqdm(prompts, desc="Processing prompts"):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            prefix = inputs.input_ids[0].tolist()

            for _ in range(num_samples_per_prompt):
                # 草稿模型生成
                with torch.no_grad():
                    draft_outputs = self.draft_model.generate(
                        inputs.input_ids,
                        max_new_tokens=max_length,
                        do_sample=True,
                        temperature=1.0,
                        output_hidden_states=True,
                        return_dict_in_generate=True,
                    )
                    draft_tokens = draft_outputs.sequences[0, len(prefix):].tolist()

                    # 目标模型生成 (用于比较)
                    target_outputs = self.target_model.generate(
                        inputs.input_ids,
                        max_new_tokens=max_length,
                        do_sample=True,
                        temperature=1.0,
                    )
                    target_tokens = target_outputs[0, len(prefix):].tolist()

                # 截断到相同长度
                min_len = min(len(draft_tokens), len(target_tokens))
                if min_len > 0:
                    sample = {
                        'prefix': prefix,
                        'tokens': target_tokens[:min_len],
                        'draft': draft_tokens[:min_len],
                    }
                    all_samples.append(sample)

        # 保存数据
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_samples, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(all_samples)} samples to {output_path}")
        return all_samples

    def train(self):
        """训练接受概率预测器"""
        print("Starting training...")

        # 加载数据
        train_dataset = AcceptanceDataset(
            self.config.train_data_path,
            self.config.mixing_ratio,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
        )

        # 优化器
        optimizer = torch.optim.AdamW(
            self.acceptance_head.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # 学习率调度器
        total_steps = len(train_loader) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # 损失函数 (带类别权重)
        weight = torch.tensor([1.0, self.config.weight_mismatch]).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=IGNORE_INDEX)

        # 训练循环
        self.acceptance_head.train()
        global_step = 0

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}")

            for batch in pbar:
                # 构建输入序列
                input_ids = torch.cat([batch['prefixes'], batch['drafts']], dim=1).to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # 获取草稿模型隐藏状态
                with torch.no_grad():
                    outputs = self.draft_model(
                        input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                    hidden_states = outputs.hidden_states[-1]

                # 只取草稿部分的隐藏状态
                prefix_len = batch['prefixes'].shape[1]
                draft_hidden = hidden_states[:, prefix_len:, :]

                # 预测接受概率
                batch_size, seq_len, hidden_dim = draft_hidden.shape
                draft_hidden_flat = draft_hidden.reshape(-1, hidden_dim)
                logits = self.acceptance_head(draft_hidden_flat)
                logits = logits.reshape(batch_size, seq_len, 2)

                # 计算损失
                logits_flat = logits.reshape(-1, 2)
                labels_flat = labels.reshape(-1)
                loss = criterion(logits_flat, labels_flat)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.acceptance_head.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                # 统计
                epoch_loss += loss.item()

                valid_mask = labels_flat != IGNORE_INDEX
                if valid_mask.sum() > 0:
                    preds = logits_flat[valid_mask].argmax(dim=-1)
                    epoch_correct += (preds == labels_flat[valid_mask]).sum().item()
                    epoch_total += valid_mask.sum().item()

                global_step += 1

                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{epoch_correct / max(epoch_total, 1):.4f}",
                })

            avg_loss = epoch_loss / len(train_loader)
            accuracy = epoch_correct / max(epoch_total, 1)
            print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")

        # 保存模型
        self._save_model()

    def _save_model(self):
        """保存模型"""
        os.makedirs(self.config.output_dir, exist_ok=True)

        # 保存模型权重
        model_path = os.path.join(self.config.output_dir, 'acceptance_head.pt')
        torch.save(self.acceptance_head.state_dict(), model_path)

        # 保存配置
        config_path = os.path.join(self.config.output_dir, 'config.json')
        config_dict = {
            'hidden_size': self.hidden_size,
            'num_layers': self.config.num_res_layers,
            'draft_model_path': self.config.draft_model_path,
            'target_model_path': self.config.target_model_path,
        }
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        print(f"Model saved to {self.config.output_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Train Acceptance Prediction Head")

    parser.add_argument("--draft_model_path", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--target_model_path", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--train_data_path", type=str, default="./data/train.json")
    parser.add_argument("--output_dir", type=str, default="./output/acceptance_head")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--num_res_layers", type=int, default=1)
    parser.add_argument("--mixing_ratio", type=float, default=0.15)
    parser.add_argument("--weight_mismatch", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bf16", action="store_true")

    args = parser.parse_args()

    config = TrainConfig(
        draft_model_path=args.draft_model_path,
        target_model_path=args.target_model_path,
        train_data_path=args.train_data_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_res_layers=args.num_res_layers,
        mixing_ratio=args.mixing_ratio,
        weight_mismatch=args.weight_mismatch,
        seed=args.seed,
        device=args.device,
        bf16=args.bf16,
    )

    trainer = AcceptanceHeadTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
