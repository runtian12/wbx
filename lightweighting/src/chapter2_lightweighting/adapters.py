from __future__ import annotations

from typing import Optional, Sequence

import torch.nn as nn

from .config import ModelTopology


class LlamaStyleAdapter:
    """
    针对 HuggingFace CausalLM 的轻量适配器。
    默认兼容 LLaMA / Mistral / Qwen2 等常见命名方式。
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.backbone = self._find_backbone(model)
        self.layers = self._find_layers(self.backbone)
        self.embed_tokens = self._find_embed_tokens(self.backbone)
        self.final_norm = self._find_final_norm(self.backbone)
        self.lm_head = getattr(model, "lm_head", None)
        self.config = model.config

    @staticmethod
    def _find_backbone(model: nn.Module) -> nn.Module:
        for attr in ["model", "transformer", "backbone"]:
            if hasattr(model, attr):
                return getattr(model, attr)
        return model

    @staticmethod
    def _find_layers(backbone: nn.Module) -> nn.ModuleList:
        for attr in ["layers", "h", "decoder_layers"]:
            if hasattr(backbone, attr):
                layers = getattr(backbone, attr)
                if isinstance(layers, (nn.ModuleList, list, tuple)):
                    return layers
        raise ValueError("未找到 Transformer 层列表，请按实际模型补充适配逻辑。")

    @staticmethod
    def _find_embed_tokens(backbone: nn.Module) -> Optional[nn.Embedding]:
        for attr in ["embed_tokens", "wte", "tok_embeddings"]:
            if hasattr(backbone, attr):
                return getattr(backbone, attr)
        return None

    @staticmethod
    def _find_final_norm(backbone: nn.Module) -> Optional[nn.Module]:
        for attr in ["norm", "final_layernorm", "ln_f"]:
            if hasattr(backbone, attr):
                return getattr(backbone, attr)
        return None

    @staticmethod
    def _find_attn_module(layer: nn.Module) -> nn.Module:
        for attr in ["self_attn", "attn", "attention"]:
            if hasattr(layer, attr):
                return getattr(layer, attr)
        raise ValueError("未找到注意力模块。")

    @staticmethod
    def _find_mlp_module(layer: nn.Module) -> nn.Module:
        for attr in ["mlp", "feed_forward", "ffn"]:
            if hasattr(layer, attr):
                return getattr(layer, attr)
        raise ValueError("未找到 MLP 模块。")

    @staticmethod
    def _pick_linear(module: nn.Module, names: Sequence[str]) -> nn.Linear:
        for name in names:
            if hasattr(module, name):
                return getattr(module, name)
        raise ValueError(f"未找到线性层，候选名称: {names}")

    def get_topology(self) -> ModelTopology:
        hidden = int(self.config.hidden_size)
        num_heads = int(getattr(self.config, "num_attention_heads", None) or getattr(self.config, "num_heads"))
        inter = int(getattr(self.config, "intermediate_size", hidden * 4))
        num_layers = int(getattr(self.config, "num_hidden_layers", len(self.layers)))
        return ModelTopology(
            num_layers=num_layers,
            hidden_size=hidden,
            intermediate_size=inter,
            num_heads=num_heads,
        )

    def iter_layer_groups(self):
        for i, layer in enumerate(self.layers):
            attn = self._find_attn_module(layer)
            mlp = self._find_mlp_module(layer)
            q_proj = self._pick_linear(attn, ["q_proj", "q_proj_linear", "query"])
            k_proj = self._pick_linear(attn, ["k_proj", "k_proj_linear", "key"])
            v_proj = self._pick_linear(attn, ["v_proj", "v_proj_linear", "value"])
            o_proj = self._pick_linear(attn, ["o_proj", "out_proj", "dense"])
            gate_proj = self._pick_linear(mlp, ["gate_proj", "fc1", "w1"])
            up_proj = self._pick_linear(mlp, ["up_proj", "fc3", "w3"])
            down_proj = self._pick_linear(mlp, ["down_proj", "fc2", "w2"])
            input_ln = getattr(layer, "input_layernorm", None) or getattr(layer, "ln_1", None)
            post_attn_ln = getattr(layer, "post_attention_layernorm", None) or getattr(layer, "ln_2", None)
            yield {
                "layer_idx": i,
                "attn": attn,
                "mlp": mlp,
                "q_proj": q_proj,
                "k_proj": k_proj,
                "v_proj": v_proj,
                "o_proj": o_proj,
                "gate_proj": gate_proj,
                "up_proj": up_proj,
                "down_proj": down_proj,
                "input_ln": input_ln,
                "post_attn_ln": post_attn_ln,
            }
