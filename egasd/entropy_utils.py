"""
熵计算和动态阈值调整工具
EGASD的核心创新点
"""

import torch
import math
from typing import Optional, Tuple


def compute_entropy(probs: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    计算概率分布的熵
    
    H = -∑ p(x) × log(p(x))
    
    Args:
        probs: 概率分布 [batch_size, vocab_size] 或 [vocab_size]
        eps: 防止log(0)的小常数
    
    Returns:
        entropy: 熵值 [batch_size] 或 标量
    """
    # 确保概率非负
    probs = torch.clamp(probs, min=eps)
    
    # 计算熵
    entropy = -torch.sum(probs * torch.log(probs), dim=-1)
    
    return entropy


def normalize_entropy(entropy: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """
    归一化熵值到[0, 1]范围
    
    H_norm = H / H_max, 其中 H_max = log(vocab_size)
    
    Args:
        entropy: 原始熵值
        vocab_size: 词表大小
    
    Returns:
        normalized_entropy: 归一化熵值 [0, 1]
    """
    h_max = math.log(vocab_size)
    return entropy / h_max


def compute_dynamic_threshold(
    entropy_norm: torch.Tensor,
    h_min: float = 0.4,
    h_max: float = 0.8,
) -> torch.Tensor:
    """
    基于归一化熵计算动态停止阈值
    
    h_stop = h_min + (h_max - h_min) × (1 - H_norm)
    
    设计逻辑:
    - 低熵(模型确定) → 高阈值 → 允许更长草稿
    - 高熵(模型不确定) → 低阈值 → 更早停止
    
    Args:
        entropy_norm: 归一化熵值 [0, 1]
        h_min: 阈值下界 (保守)
        h_max: 阈值上界 (激进)
    
    Returns:
        dynamic_threshold: 动态阈值
    """
    return h_min + (h_max - h_min) * (1 - entropy_norm)


def compute_cumulative_rejection_prob(
    acceptance_probs: torch.Tensor
) -> torch.Tensor:
    """
    计算累计拒绝概率
    
    P_reject_cum = 1 - ∏ P_accept_i
    
    Args:
        acceptance_probs: 各token的接受概率 [seq_len]
    
    Returns:
        cumulative_rejection_prob: 累计拒绝概率
    """
    cumulative_acceptance = torch.prod(acceptance_probs)
    return 1 - cumulative_acceptance


def should_stop_drafting(
    cumulative_rejection_prob: float,
    dynamic_threshold: float,
    current_length: int,
    max_length: int,
    min_length: int = 1,
) -> Tuple[bool, str]:
    """
    判断是否应该停止草稿生成
    
    Args:
        cumulative_rejection_prob: 累计拒绝概率
        dynamic_threshold: 动态阈值
        current_length: 当前草稿长度
        max_length: 最大草稿长度
        min_length: 最小草稿长度
    
    Returns:
        should_stop: 是否停止
        reason: 停止原因
    """
    # 未达到最小长度，继续生成
    if current_length < min_length:
        return False, "below_min_length"
    
    # 达到最大长度，停止
    if current_length >= max_length:
        return True, "max_length_reached"
    
    # 累计拒绝概率超过动态阈值，停止
    if cumulative_rejection_prob > dynamic_threshold:
        return True, "threshold_exceeded"
    
    return False, "continue"


class EntropyGuidedThresholdManager:
    """
    熵引导的动态阈值管理器
    
    封装了熵计算、归一化和动态阈值调整的完整流程
    """
    
    def __init__(
        self,
        vocab_size: int,
        h_min: float = 0.4,
        h_max: float = 0.8,
        min_draft_length: int = 2,
        max_draft_length: int = 20,
    ):
        """
        Args:
            vocab_size: 词表大小
            h_min: 阈值下界
            h_max: 阈值上界
            min_draft_length: 最小草稿长度
            max_draft_length: 最大草稿长度
        """
        self.vocab_size = vocab_size
        self.h_min = h_min
        self.h_max = h_max
        self.min_draft_length = min_draft_length
        self.max_draft_length = max_draft_length
        self.h_max_value = math.log(vocab_size)
        
        # 统计信息
        self.entropy_history = []
        self.threshold_history = []
        self.draft_length_history = []
    
    def compute_threshold(self, probs: torch.Tensor) -> Tuple[float, float, float]:
        """
        根据概率分布计算动态阈值
        
        Args:
            probs: 草稿模型的输出概率分布
        
        Returns:
            dynamic_threshold: 动态阈值
            entropy: 原始熵值
            entropy_norm: 归一化熵值
        """
        entropy = compute_entropy(probs)
        entropy_norm = normalize_entropy(entropy, self.vocab_size)
        dynamic_threshold = compute_dynamic_threshold(
            entropy_norm, self.h_min, self.h_max
        )
        
        # 记录历史
        self.entropy_history.append(entropy.item() if torch.is_tensor(entropy) else entropy)
        self.threshold_history.append(
            dynamic_threshold.item() if torch.is_tensor(dynamic_threshold) else dynamic_threshold
        )
        
        return (
            dynamic_threshold.item() if torch.is_tensor(dynamic_threshold) else dynamic_threshold,
            entropy.item() if torch.is_tensor(entropy) else entropy,
            entropy_norm.item() if torch.is_tensor(entropy_norm) else entropy_norm,
        )
    
    def should_stop(
        self,
        cumulative_rejection_prob: float,
        dynamic_threshold: float,
        current_length: int,
    ) -> Tuple[bool, str]:
        """
        判断是否停止草稿生成
        """
        return should_stop_drafting(
            cumulative_rejection_prob,
            dynamic_threshold,
            current_length,
            self.max_draft_length,
            self.min_draft_length,
        )
    
    def record_draft_length(self, length: int):
        """记录草稿长度"""
        self.draft_length_history.append(length)
    
    def get_statistics(self) -> dict:
        """获取统计信息"""
        import numpy as np
        
        return {
            'avg_entropy': np.mean(self.entropy_history) if self.entropy_history else 0,
            'avg_threshold': np.mean(self.threshold_history) if self.threshold_history else 0,
            'avg_draft_length': np.mean(self.draft_length_history) if self.draft_length_history else 0,
            'total_iterations': len(self.draft_length_history),
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.entropy_history = []
        self.threshold_history = []
        self.draft_length_history = []
