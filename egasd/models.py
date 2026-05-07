"""
EGASD模型组件
包含接受概率预测器和Pivot分类器
"""

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin


class ResBlock(nn.Module):
    """残差块，用于接受概率预测器"""
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.linear(x))


class AcceptancePredictionHead(nn.Module, PyTorchModelHubMixin):
    """
    接受概率预测器 (来自SpecDec++)
    
    使用草稿模型的最后一层隐藏状态预测token被目标模型接受的概率
    
    Args:
        config: 配置字典，包含:
            - hidden_size: 隐藏层维度
            - num_layers: 残差块数量
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_size = config['hidden_size']
        num_layers = config.get('num_layers', 1)
        
        self.model = nn.Sequential(
            *[ResBlock(hidden_size) for _ in range(num_layers)],
            nn.Linear(hidden_size, 2),
        )

    def forward(self, x):
        """
        Args:
            x: 草稿模型的隐藏状态 [batch_size, hidden_size]
        Returns:
            logits: 接受/拒绝的logits [batch_size, 2]
        """
        return self.model(x)
    
    def predict_acceptance_prob(self, hidden_state):
        """
        预测接受概率
        
        Args:
            hidden_state: 草稿模型的隐藏状态
        Returns:
            acceptance_prob: 接受概率 [0, 1]
        """
        logits = self.forward(hidden_state)
        probs = torch.softmax(logits, dim=-1)
        return probs[..., 1]  # 返回接受类的概率


class PivotClassifier(nn.Module):
    """
    Pivot分类器 (来自PAD)
    
    识别关键转折token，用于差异化验证策略
    
    Args:
        target_hidden_dim: 目标模型隐藏层维度
        t_embed: 目标隐藏状态嵌入维度
        s_embed: 小特征嵌入维度
        hidden_dim: 分类器隐藏层维度
    """
    def __init__(
        self, 
        target_hidden_dim=4096, 
        t_embed=64, 
        s_embed=16, 
        hidden_dim=128,
        **kwargs
    ):
        super().__init__()
        
        # 目标模型隐藏状态投影
        self.target_proj = nn.Linear(target_hidden_dim, t_embed)
        
        # 小特征投影 (target_entropy, target_logit)
        small_features_dim = 2
        self.small_features_proj = nn.Linear(small_features_dim, s_embed)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(t_embed + s_embed, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        target_hidden,
        target_entropy,
        target_logit,
        **kwargs
    ):
        """
        Args:
            target_hidden: 目标模型隐藏状态 [batch_size, seq_len, hidden_dim]
            target_entropy: 目标模型预测熵 [batch_size, seq_len, 1]
            target_logit: 目标模型对选中token的概率 [batch_size, seq_len, 1]
        Returns:
            logits: Pivot分类logits [batch_size, seq_len, 2]
        """
        t_embed = self.target_proj(target_hidden)
        small_features = torch.cat([target_entropy, target_logit], dim=-1)
        s_embed = self.small_features_proj(small_features)
        combined = torch.cat([t_embed, s_embed], dim=-1)
        logits = self.classifier(combined)
        return logits
    
    def predict_pivot_score(self, target_hidden, target_entropy, target_logit):
        """
        预测Pivot分数
        
        Args:
            target_hidden: 目标模型隐藏状态
            target_entropy: 目标模型预测熵
            target_logit: 目标模型对选中token的概率
        Returns:
            pivot_score: Pivot分数 [0, 1]
        """
        logits = self.forward(target_hidden, target_entropy, target_logit)
        probs = torch.softmax(logits, dim=-1)
        return probs[..., 1]  # 返回Pivot类的概率


class EGASDModel(nn.Module):
    """
    EGASD完整模型封装
    
    整合接受概率预测器和Pivot分类器
    """
    def __init__(
        self,
        draft_hidden_size,
        target_hidden_size,
        acc_head_num_layers=1,
        pivot_t_embed=64,
        pivot_s_embed=16,
        pivot_hidden_dim=128,
    ):
        super().__init__()
        
        # 接受概率预测器
        acc_config = {
            'hidden_size': draft_hidden_size,
            'num_layers': acc_head_num_layers
        }
        self.acceptance_head = AcceptancePredictionHead(acc_config)
        
        # Pivot分类器
        self.pivot_classifier = PivotClassifier(
            target_hidden_dim=target_hidden_size,
            t_embed=pivot_t_embed,
            s_embed=pivot_s_embed,
            hidden_dim=pivot_hidden_dim,
        )
    
    def save_pretrained(self, save_dir):
        """保存模型"""
        import os
        import json
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存接受概率预测器
        acc_dir = os.path.join(save_dir, 'acceptance_head')
        os.makedirs(acc_dir, exist_ok=True)
        torch.save(self.acceptance_head.state_dict(), os.path.join(acc_dir, 'model.pt'))
        with open(os.path.join(acc_dir, 'config.json'), 'w') as f:
            json.dump(self.acceptance_head.config, f)
        
        # 保存Pivot分类器
        pivot_dir = os.path.join(save_dir, 'pivot_classifier')
        os.makedirs(pivot_dir, exist_ok=True)
        torch.save(self.pivot_classifier.state_dict(), os.path.join(pivot_dir, 'model.pt'))
    
    @classmethod
    def from_pretrained(cls, load_dir, draft_hidden_size, target_hidden_size):
        """加载模型"""
        import os
        import json
        
        model = cls(draft_hidden_size, target_hidden_size)
        
        # 加载接受概率预测器
        acc_dir = os.path.join(load_dir, 'acceptance_head')
        model.acceptance_head.load_state_dict(
            torch.load(os.path.join(acc_dir, 'model.pt'))
        )
        
        # 加载Pivot分类器
        pivot_dir = os.path.join(load_dir, 'pivot_classifier')
        model.pivot_classifier.load_state_dict(
            torch.load(os.path.join(pivot_dir, 'model.pt'))
        )
        
        return model
