"""
EGASD推测解码主函数
基于熵引导的自适应推测解码
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Union
from dataclasses import dataclass
import time

try:
    from .entropy_utils import (
        compute_entropy,
        normalize_entropy,
        compute_dynamic_threshold,
        EntropyGuidedThresholdManager,
    )
    from .models import AcceptancePredictionHead, PivotClassifier
except ImportError:
    from entropy_utils import (
        compute_entropy,
        normalize_entropy,
        compute_dynamic_threshold,
        EntropyGuidedThresholdManager,
    )
    from models import AcceptancePredictionHead, PivotClassifier


@dataclass
class EGASDConfig:
    """EGASD配置"""
    # 动态阈值参数
    h_min: float = 0.4  # 阈值下界
    h_max: float = 0.8  # 阈值上界
    
    # 草稿长度限制
    min_draft_length: int = 2
    max_draft_length: int = 20
    
    # Pivot验证参数
    pivot_threshold: float = 0.5  # Pivot阈值σ
    relaxation_factor: float = 1.2  # 宽松验证的放宽因子
    
    # 采样参数
    do_sample: bool = True
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    
    # 其他
    max_new_tokens: int = 512
    use_pivot_verification: bool = True  # 是否使用Pivot验证


class EGASDDecoder:
    """
    EGASD解码器
    
    整合了:
    1. 熵引导的动态阈值调整 (本文创新点)
    2. SpecDec++的接受概率预测
    3. PAD的Pivot-Aware验证
    """
    
    def __init__(
        self,
        target_model: nn.Module,
        draft_model: nn.Module,
        tokenizer,
        acceptance_head: Optional[AcceptancePredictionHead] = None,
        pivot_classifier: Optional[PivotClassifier] = None,
        config: Optional[EGASDConfig] = None,
        device: str = "cuda",
    ):
        """
        Args:
            target_model: 目标模型 (大模型)
            draft_model: 草稿模型 (小模型)
            tokenizer: 分词器
            acceptance_head: 接受概率预测器
            pivot_classifier: Pivot分类器
            config: EGASD配置
            device: 设备
        """
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.acceptance_head = acceptance_head
        self.pivot_classifier = pivot_classifier
        self.config = config or EGASDConfig()
        self.device = device
        
        # 获取词表大小
        self.vocab_size = target_model.config.vocab_size
        
        # 初始化阈值管理器
        self.threshold_manager = EntropyGuidedThresholdManager(
            vocab_size=self.vocab_size,
            h_min=self.config.h_min,
            h_max=self.config.h_max,
            min_draft_length=self.config.min_draft_length,
            max_draft_length=self.config.max_draft_length,
        )
        
        # 统计信息
        self.stats = {
            'total_tokens': 0,
            'accepted_tokens': 0,
            'draft_iterations': 0,
            'total_draft_tokens': 0,
            'inference_time': 0,
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_tokens': 0,
            'accepted_tokens': 0,
            'draft_iterations': 0,
            'total_draft_tokens': 0,
            'inference_time': 0,
        }
        self.threshold_manager.reset_statistics()
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """
        EGASD生成函数
        
        Args:
            input_ids: 输入token ids [batch_size, seq_len]
            attention_mask: 注意力掩码
            max_new_tokens: 最大生成token数
        
        Returns:
            generated_ids: 生成的token ids
            stats: 统计信息
        """
        start_time = time.time()
        
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        batch_size = input_ids.shape[0]
        assert batch_size == 1, "目前仅支持batch_size=1"
        
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        generated_ids = input_ids.clone()
        init_len = input_ids.shape[1]
        
        # KV缓存
        draft_past_kv = None
        target_past_kv = None
        
        eos_token_id = self.tokenizer.eos_token_id
        
        while generated_ids.shape[1] - init_len < max_new_tokens:
            # ========== 草稿生成阶段 ==========
            draft_tokens, draft_probs, draft_hidden_states = self._draft_generation(
                generated_ids,
                draft_past_kv,
            )
            
            if len(draft_tokens) == 0:
                # 无草稿token，直接从目标模型采样
                new_token = self._sample_from_target(generated_ids, target_past_kv)
                generated_ids = torch.cat([generated_ids, new_token.unsqueeze(0).unsqueeze(0)], dim=1)
                
                if new_token.item() == eos_token_id:
                    break
                continue
            
            self.stats['draft_iterations'] += 1
            self.stats['total_draft_tokens'] += len(draft_tokens)
            
            # ========== 验证阶段 ==========
            accepted_tokens, n_accepted = self._verification(
                generated_ids,
                draft_tokens,
                draft_probs,
                target_past_kv,
            )
            
            self.stats['accepted_tokens'] += n_accepted
            
            # 更新生成序列
            generated_ids = torch.cat([
                generated_ids,
                accepted_tokens.unsqueeze(0)
            ], dim=1)
            
            # 检查EOS
            if eos_token_id in accepted_tokens.tolist():
                break
            
            # 更新KV缓存位置
            draft_past_kv = generated_ids.shape[1] - 2
            target_past_kv = generated_ids.shape[1] - 2
        
        self.stats['total_tokens'] = generated_ids.shape[1] - init_len
        self.stats['inference_time'] = time.time() - start_time
        
        # 合并统计信息
        stats = {
            **self.stats,
            **self.threshold_manager.get_statistics(),
            'acceptance_rate': self.stats['accepted_tokens'] / max(self.stats['total_draft_tokens'], 1),
            'tokens_per_iteration': self.stats['total_tokens'] / max(self.stats['draft_iterations'], 1),
        }
        
        return generated_ids, stats
    
    def _draft_generation(
        self,
        input_ids: torch.Tensor,
        past_kv: Optional[int] = None,
    ) -> Tuple[List[int], List[torch.Tensor], List[torch.Tensor]]:
        """
        草稿生成阶段
        
        使用熵引导的动态阈值决定何时停止
        
        Returns:
            draft_tokens: 草稿token列表
            draft_probs: 草稿概率分布列表
            draft_hidden_states: 草稿隐藏状态列表
        """
        draft_tokens = []
        draft_probs = []
        draft_hidden_states = []
        
        cumulative_rejection_prob = 0.0
        current_input = input_ids.clone()
        
        for i in range(self.config.max_draft_length):
            # 草稿模型前向传播
            outputs = self.draft_model(
                current_input,
                output_hidden_states=True,
                use_cache=True,
            )
            
            logits = outputs.logits[:, -1, :]
            hidden_state = outputs.hidden_states[-1][:, -1, :]
            
            # 计算概率分布
            if self.config.do_sample:
                probs = self._apply_sampling(logits)
            else:
                probs = torch.softmax(logits, dim=-1)
            
            # ========== 熵引导的动态阈值 (核心创新) ==========
            dynamic_threshold, entropy, entropy_norm = self.threshold_manager.compute_threshold(probs[0])
            
            # 预测接受概率
            if self.acceptance_head is not None:
                acceptance_prob = self.acceptance_head.predict_acceptance_prob(hidden_state[0])
                acceptance_prob = acceptance_prob.item()
            else:
                # 如果没有接受概率预测器，使用启发式方法
                acceptance_prob = 1.0 - entropy_norm
            
            # 更新累计拒绝概率
            if i == 0:
                cumulative_rejection_prob = 0.0
            else:
                cumulative_rejection_prob = 1 - (1 - cumulative_rejection_prob) * acceptance_prob
            
            # 判断是否停止
            should_stop, reason = self.threshold_manager.should_stop(
                cumulative_rejection_prob,
                dynamic_threshold,
                len(draft_tokens),
            )
            
            if should_stop:
                break
            
            # 采样token
            if self.config.do_sample:
                token = torch.multinomial(probs, num_samples=1)
            else:
                token = logits.argmax(dim=-1, keepdim=True)
            
            draft_tokens.append(token.item())
            draft_probs.append(probs)
            draft_hidden_states.append(hidden_state)
            
            # 更新输入
            current_input = torch.cat([current_input, token], dim=1)
            
            # 检查EOS
            if token.item() == self.tokenizer.eos_token_id:
                break
        
        self.threshold_manager.record_draft_length(len(draft_tokens))
        
        return draft_tokens, draft_probs, draft_hidden_states
    
    def _verification(
        self,
        input_ids: torch.Tensor,
        draft_tokens: List[int],
        draft_probs: List[torch.Tensor],
        past_kv: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        验证阶段
        
        使用Pivot-Aware验证策略
        
        Returns:
            accepted_tokens: 接受的token
            n_accepted: 接受的token数量
        """
        n_draft = len(draft_tokens)
        
        # 构建候选序列
        draft_tensor = torch.tensor(draft_tokens, device=self.device).unsqueeze(0)
        candidate_ids = torch.cat([input_ids, draft_tensor], dim=1)
        
        # 目标模型前向传播
        outputs = self.target_model(
            candidate_ids,
            output_hidden_states=True,
            use_cache=True,
        )
        
        target_logits = outputs.logits[:, -n_draft-1:, :]
        target_hidden = outputs.hidden_states[-1][:, -n_draft:, :]
        
        # 计算目标模型概率分布
        if self.config.do_sample:
            target_probs = self._apply_sampling(target_logits)
        else:
            target_probs = torch.softmax(target_logits, dim=-1)
        
        # 验证每个草稿token
        accepted_tokens = []
        
        for i in range(n_draft):
            draft_token = draft_tokens[i]
            p = target_probs[0, i, :]  # 目标模型概率
            q = draft_probs[i][0]  # 草稿模型概率
            
            # 计算Pivot分数 (如果启用)
            if self.config.use_pivot_verification and self.pivot_classifier is not None:
                target_entropy = compute_entropy(p).unsqueeze(-1).unsqueeze(0)
                target_logit = p[draft_token].unsqueeze(-1).unsqueeze(0)
                
                pivot_score = self.pivot_classifier.predict_pivot_score(
                    target_hidden[:, i:i+1, :],
                    target_entropy,
                    target_logit,
                )
                pivot_score = pivot_score.item()
                
                # 根据Pivot分数选择验证策略
                if pivot_score > self.config.pivot_threshold:
                    # 严格验证
                    accept = self._standard_rejection_sampling(p, q, draft_token)
                else:
                    # 宽松验证
                    accept = self._relaxed_verification(p, q, draft_token)
            else:
                # 标准拒绝采样
                accept = self._standard_rejection_sampling(p, q, draft_token)
            
            if accept:
                accepted_tokens.append(draft_token)
            else:
                # 从残差分布采样修正token
                residual = torch.clamp(p - q, min=0)
                if residual.sum() > 0:
                    residual = residual / residual.sum()
                    correction_token = torch.multinomial(residual, num_samples=1).item()
                else:
                    correction_token = torch.multinomial(p, num_samples=1).item()
                accepted_tokens.append(correction_token)
                break
        
        # 如果所有草稿token都被接受，额外采样一个token
        if len(accepted_tokens) == n_draft:
            bonus_token = torch.multinomial(target_probs[0, -1, :], num_samples=1).item()
            accepted_tokens.append(bonus_token)
        
        return torch.tensor(accepted_tokens, device=self.device), len(accepted_tokens) - 1
    
    def _standard_rejection_sampling(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        token: int,
    ) -> bool:
        """标准拒绝采样"""
        p_token = p[token].item()
        q_token = q[token].item()
        
        if q_token == 0:
            return p_token > 0
        
        acceptance_prob = min(1.0, p_token / q_token)
        r = torch.rand(1).item()
        
        return r < acceptance_prob
    
    def _relaxed_verification(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        token: int,
    ) -> bool:
        """宽松验证 (用于非Pivot位置)"""
        p_token = p[token].item()
        q_token = q[token].item()
        
        if q_token == 0:
            return p_token > 0
        
        # 使用放宽因子
        acceptance_prob = min(1.0, self.config.relaxation_factor * p_token / q_token)
        r = torch.rand(1).item()
        
        return r < acceptance_prob
    
    def _sample_from_target(
        self,
        input_ids: torch.Tensor,
        past_kv: Optional[int] = None,
    ) -> torch.Tensor:
        """从目标模型采样单个token"""
        outputs = self.target_model(input_ids, use_cache=True)
        logits = outputs.logits[:, -1, :]
        
        if self.config.do_sample:
            probs = self._apply_sampling(logits)
            token = torch.multinomial(probs, num_samples=1)
        else:
            token = logits.argmax(dim=-1, keepdim=True)
        
        return token[0, 0]
    
    def _apply_sampling(self, logits: torch.Tensor) -> torch.Tensor:
        """应用采样策略 (temperature, top-k, top-p)"""
        logits = logits / self.config.temperature
        
        if self.config.top_k is not None:
            top_k = min(self.config.top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        if self.config.top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > self.config.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        return torch.softmax(logits, dim=-1)


def egasd_generate(
    target_model: nn.Module,
    draft_model: nn.Module,
    tokenizer,
    input_ids: torch.Tensor,
    acceptance_head: Optional[AcceptancePredictionHead] = None,
    pivot_classifier: Optional[PivotClassifier] = None,
    config: Optional[EGASDConfig] = None,
    **kwargs,
) -> Tuple[torch.Tensor, dict]:
    """
    EGASD生成的便捷函数
    
    Args:
        target_model: 目标模型
        draft_model: 草稿模型
        tokenizer: 分词器
        input_ids: 输入token ids
        acceptance_head: 接受概率预测器
        pivot_classifier: Pivot分类器
        config: EGASD配置
    
    Returns:
        generated_ids: 生成的token ids
        stats: 统计信息
    """
    decoder = EGASDDecoder(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        acceptance_head=acceptance_head,
        pivot_classifier=pivot_classifier,
        config=config,
    )
    
    return decoder.generate(input_ids, **kwargs)
