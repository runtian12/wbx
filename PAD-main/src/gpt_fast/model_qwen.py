# Qwen2 and Qwen3 model implementations for gpt-fast
# 
# Key differences from LLaMA (model.py):
# 
# 1. **Decoupled head_dim**:
#    - Qwen models have head_dim explicitly specified, not calculated as dim // n_head
#    - This means n_head * head_dim may be larger than dim (hidden_size)
#    - Example: Qwen3-32B has dim=5120, n_head=64, head_dim=128 → 64×128=8192 > 5120
# 
# 2. **QKV Bias**:
#    - Qwen2: bias=True in QKV projections (LLaMA uses bias=False)
#    - Qwen3: bias=False in QKV projections (same as LLaMA)
# 
# 3. **QK Normalization** (Qwen3 only):
#    - Adds RMSNorm on Q and K tensors before applying rotary embeddings
#    - This is the unique Qwen3 feature that distinguishes it from Qwen2
# 
# 4. **Unified Implementation**:
#    - Single QwenTransformer class handles both Qwen2 and Qwen3
#    - Model variant controlled by qkv_bias and use_qk_norm parameters

from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor

# Import all common components from the base model
from .model import (
    find_multiple, ModelArgs, KVCache, RMSNorm, FeedForward, 
    TransformerBlock, Transformer, Attention, apply_rope_scaling, 
    precompute_freqs_cis, apply_rotary_emb, get_mask_mod
)
from torch.nn.attention.flex_attention import (
    BlockMask,
    flex_attention,
)


# flex_attention = torch.compile(flex_attention, fullgraph=True, mode="max-autotune")

# =============================================================================
# UNIFIED QWEN IMPLEMENTATION (Qwen2 + Qwen3)
# =============================================================================

@dataclass
class QwenModelArgs(ModelArgs):
    """Unified Qwen model arguments for both Qwen2 and Qwen3"""
    qkv_bias: bool = True       # True for Qwen2, False for Qwen3
    use_qk_norm: bool = False   # False for Qwen2, True for Qwen3
    
    def __post_init__(self):
        # Store the explicitly provided head_dim before calling parent
        explicit_head_dim = self.head_dim
        
        # Call parent __post_init__ for other initialization
        super().__post_init__()
        
        # Restore the explicit head_dim for Qwen models (don't auto-calculate)
        # This is crucial because Qwen models have head_dim decoupled from hidden_size
        self.head_dim = explicit_head_dim
    
    @classmethod
    def from_name(cls, name: str):
        if name in qwen_configs:
            return cls(**qwen_configs[name])
        # fuzzy search
        config = [config for config in qwen_configs if config.lower() in str(name).lower()]

        # We may have two or more configs matched (e.g. "7B" and "Mistral-7B"). Find the best config match,
        # take longer name (as it have more symbols matched)
        if len(config) > 1:
            config.sort(key=len, reverse=True)
            assert len(config[0]) != len(config[1]), name # make sure only one 'best' match
        return cls(**qwen_configs[config[0]])
            


# Unified Qwen configurations - covers both Qwen2 and Qwen3
# 
# IMPORTANT: For Qwen models, head_dim is explicitly specified and decoupled from hidden_size.
# This means n_head * head_dim may be larger than dim (hidden_size).
# For example, Qwen3-32B: dim=5120, n_head=64, head_dim=128 → 64×128=8192 > 5120
qwen_configs = {
    # Qwen2 models (qkv_bias=True, use_qk_norm=False)
    "qwen2-0.5b": dict(
        block_size=32768, n_layer=24, n_head=14, n_local_heads=2, dim=896,
        intermediate_size=4864, vocab_size=151936, rope_base=1000000,
        norm_eps=1e-6, head_dim=64, qkv_bias=True, use_qk_norm=False
    ),
    "qwen2-1.5b": dict(
        block_size=32768, n_layer=28, n_head=12, n_local_heads=2, dim=1536,
        intermediate_size=8960, vocab_size=151936, rope_base=1000000,
        norm_eps=1e-6, head_dim=128, qkv_bias=True, use_qk_norm=False
    ),
    "qwen2.5-1.5b": dict(  # Same as qwen2-1.5b but with updated vocab
        block_size=32768, n_layer=28, n_head=12, n_local_heads=2, dim=1536,
        intermediate_size=8960, vocab_size=151936, rope_base=1000000,
        norm_eps=1e-6, head_dim=128, qkv_bias=True, use_qk_norm=False
    ),
    "qwen2-7b": dict(
        block_size=32768, n_layer=28, n_head=28, n_local_heads=4, dim=3584,
        intermediate_size=18944, vocab_size=152064, rope_base=1000000,
        norm_eps=1e-6, head_dim=128, qkv_bias=True, use_qk_norm=False
    ),
    "qwen2-72b": dict(
        block_size=32768, n_layer=80, n_head=64, n_local_heads=8, dim=8192,
        intermediate_size=29568, vocab_size=152064, rope_base=1000000,
        norm_eps=1e-6, head_dim=128, qkv_bias=True, use_qk_norm=False
    ),
    
    # Qwen3 models (qkv_bias=False, use_qk_norm=True)
    "qwen3-0.6b": dict(
        block_size=40960, n_layer=28, n_head=16, n_local_heads=8, dim=1024,
        intermediate_size=3072, vocab_size=151936, rope_base=1000000,
        norm_eps=1e-6, head_dim=128, qkv_bias=False, use_qk_norm=True
    ),
    "qwen3-1.7b": dict(
        block_size=40960, n_layer=28, n_head=16, n_local_heads=8, dim=2048,
        intermediate_size=6144, vocab_size=151936, rope_base=1000000,
        norm_eps=1e-6, head_dim=128, qkv_bias=False, use_qk_norm=True
    ),
    "qwen3-8b": dict(
        block_size=40960, n_layer=36, n_head=32, n_local_heads=8, dim=4096,
        intermediate_size=12288, vocab_size=151936, rope_base=1000000,
        norm_eps=1e-6, head_dim=128, qkv_bias=False, use_qk_norm=True
    ),
    "qwen3-32b": dict(
        block_size=40960, n_layer=64, n_head=64, n_local_heads=8, dim=5120,
        intermediate_size=25600, vocab_size=151936, rope_base=1000000,
        norm_eps=1e-6, head_dim=128, qkv_bias=False, use_qk_norm=True
    ),
}


class QwenAttention(Attention):
    """Unified Qwen attention supporting both Qwen2 (bias) and Qwen3 (QK norm)"""
    
    def __init__(self, config: QwenModelArgs):
        # Initialize base attention first
        super().__init__(config)
        
        # Configure QKV projection with appropriate bias setting
        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=config.qkv_bias)
        
        # Override output projection for Qwen models (input size = n_head * head_dim, not dim)
        q_proj_size = config.n_head * config.head_dim
        self.wo = nn.Linear(q_proj_size, config.dim, bias=False)
        
        # Add QK normalization layers if needed (Qwen3 feature)
        if config.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=getattr(config, 'norm_eps', 1e-5))
            self.k_norm = RMSNorm(self.head_dim, eps=getattr(config, 'norm_eps', 1e-5))
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: BlockMask, input_pos: Optional[Tensor] = None) -> Tensor:
        if x.dim() == 2:
            # During decode_one_token, input is [batch_size, 1] but gets flattened to [batch_size]
            bsz = x.shape[0]
            seqlen = 1
            x = x.unsqueeze(1)  # Add sequence dimension back
        else:
            bsz, seqlen, _ = x.shape

        # For Qwen models, head_dim is decoupled from hidden_size
        q_size = self.n_head * self.head_dim
        kv_size = self.n_local_heads * self.head_dim

        q, k, v = self.wqkv(x).split([q_size, kv_size, kv_size], dim=-1)

        # Reshape to heads - for Qwen3, weights are now in HuggingFace layout
        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        # Apply QK normalization if enabled (Qwen3 feature)
        # Can apply directly since weights are in HuggingFace layout for Qwen3
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
            
            # For Qwen3: Permute Q/K before RoPE (since weights are unpermuted)
            def permute_for_rope(act):
                """Permute activations for RoPE compatibility"""
                bsz, seqlen, n_head, head_dim = act.shape
                half_dim = head_dim // 2
                
                # Reshape to separate first/second halves: [bsz, seqlen, n_head, 2, half_dim]
                reshaped = act.view(bsz, seqlen, n_head, 2, half_dim)
                
                # Transpose to interleave: [bsz, seqlen, n_head, half_dim, 2]  
                transposed = reshaped.transpose(-2, -1)
                
                # Flatten to get interleaved layout: [bsz, seqlen, n_head, head_dim]
                return transposed.reshape(bsz, seqlen, n_head, head_dim)
            
            q = permute_for_rope(q)
            k = permute_for_rope(k)

        # Apply RoPE
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # Transpose to attention layout
        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        # Update cache if present
        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        # Scaled dot product attention using FlexAttention
        y = flex_attention(q, k, v, block_mask=mask, enable_gqa=(self.n_head != self.n_local_heads))

        # Reshape back and apply output projection
        q_proj_size = self.n_head * self.head_dim
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, q_proj_size)
        
        result = self.wo(y)
        
        # If we added sequence dimension, remove it
        if result.shape[1] == 1 and seqlen == 1:
            result = result.squeeze(1)
        
        return result


class QwenTransformerBlock(TransformerBlock):
    """Unified Qwen transformer block"""
    
    def __init__(self, config: QwenModelArgs) -> None:
        # Call parent init but replace attention with QwenAttention
        super().__init__(config)
        self.attention = QwenAttention(config)


class QwenTransformer(Transformer):
    """Unified Qwen Transformer supporting both Qwen2 and Qwen3 variants"""
    
    def __init__(self, config: QwenModelArgs) -> None:
        # Initialize base transformer
        super().__init__(config)
        
        # Override layers with Qwen-specific blocks
        self.layers = nn.ModuleList(QwenTransformerBlock(config) for _ in range(config.n_layer))
        self.get_mask_mod = get_mask_mod

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Custom load_state_dict to handle weight layout optimization for Qwen3"""
        # For Qwen3 models with QK normalization, unpermute the QKV weights 
        # so we can apply QK norm directly and only permute before RoPE
        if self.config.use_qk_norm:  # Qwen3 marker
            state_dict = self._unpermute_qkv_weights_for_qwen3(state_dict)
        
        # Call parent load_state_dict
        if assign:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        else:
            return super().load_state_dict(state_dict, strict=strict)
    
    def _unpermute_qkv_weights_for_qwen3(self, state_dict):
        """Unpermute QKV weights to HuggingFace layout for efficient QK normalization"""
        print("Unpermuting QKV weights for efficient Qwen3 QK normalization...")
        
        def unpermute_weight(weight, n_head, head_dim):
            """Unpermute a single Q or K weight matrix from RoPE layout to HF layout"""
            dim = weight.shape[1]  # input dimension
            # Reverse the RoPE permutation: interleaved -> concatenated halves
            # weight: [n_head * head_dim, dim] -> [n_head, head_dim, dim]
            weight_heads = weight.view(n_head, head_dim, dim)
            
            # The RoPE permutation interleaves first/second halves
            # De-interleave: [a0, b0, a1, b1, ...] -> [a0, a1, ..., b0, b1, ...]
            interleaved = weight_heads.view(n_head, head_dim//2, 2, dim)
            first_half = interleaved[:, :, 0, :]   # [n_head, head_dim//2, dim]
            second_half = interleaved[:, :, 1, :]  # [n_head, head_dim//2, dim]
            
            # Concatenate back to HF layout: [first_half, second_half]
            hf_layout = torch.cat([first_half, second_half], dim=1)  # [n_head, head_dim, dim]
            
            return hf_layout.view(n_head * head_dim, dim)
        
        # Create a copy to avoid modifying the original
        new_state_dict = {}
        for key, value in state_dict.items():
            if "layers" in key and "attention.wqkv.weight" in key:
                # Extract Q, K, V from the combined weight
                q_size = self.config.n_head * self.config.head_dim
                kv_size = self.config.n_local_heads * self.config.head_dim
                
                q_weight = value[:q_size, :]
                k_weight = value[q_size:q_size+kv_size, :]
                v_weight = value[q_size+kv_size:, :]
                
                # Unpermute Q and K weights (V doesn't need permutation)
                q_unpermuted = unpermute_weight(q_weight, self.config.n_head, self.config.head_dim)
                k_unpermuted = unpermute_weight(k_weight, self.config.n_local_heads, self.config.head_dim)
                
                # Recombine the weights
                new_value = torch.cat([q_unpermuted, k_unpermuted, v_weight], dim=0)
                new_state_dict[key] = new_value
            else:
                new_state_dict[key] = value
        
        print("✓ QKV weights unpermuted for efficient QK normalization")
        return new_state_dict

    @classmethod
    def from_name(cls, name: str):
        """Create QwenTransformer with QwenModelArgs"""
        return cls(QwenModelArgs.from_name(name))

    def setup_caches(self, max_batch_size, max_seq_length):
        """Override setup_caches to use explicit head_dim for Qwen models"""
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        
        # Use explicit head_dim from config instead of calculating from dim // n_head
        head_dim = self.config.head_dim
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        dtype = self.output.weight.dtype
        
        # For quantized layers, dtype is encoded in scales
        if hasattr(self.output, "scales"):
            dtype = self.output.scales.dtype
        elif hasattr(self.output, "scales_and_zeros"):
            dtype = self.output.scales_and_zeros.dtype
            
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_local_heads, head_dim, dtype)

        # Use explicit head_dim for freqs_cis computation
        self.freqs_cis = precompute_freqs_cis(self.config.block_size, head_dim, self.config.rope_base, dtype, self.config.rope_scaling)
        # self.causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))
    
    @classmethod
    def from_name(cls, name: str):
        return cls(QwenModelArgs.from_name(name))


# Backward compatibility aliases
QwenDense = QwenTransformer  # Alternative name as suggested
Qwen2Transformer = QwenTransformer  # For existing code
Qwen3Transformer = QwenTransformer  # For existing code
