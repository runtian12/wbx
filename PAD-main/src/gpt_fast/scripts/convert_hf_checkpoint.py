# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Optional
from safetensors.torch import load_file as load_safetensors_file
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from ..gpt_fast_utils import detect_model_type, get_model_cls



def get_model_config(model_name: str, model_type: str):
    """Get appropriate model configuration based on model type"""
    if model_type in ["qwen2", "qwen3"]:
        # Import Qwen config
        from ..model_qwen import QwenModelArgs
        return QwenModelArgs.from_name(model_name)
    else:
        from ..model import ModelArgs
        # Use standard LLaMA config
        return ModelArgs.from_name(model_name)


def get_weight_mapping() -> dict:
    """Get unified weight mapping for all model types"""
    # Unified mapping that works for all models (LLaMA, Qwen2, Qwen3)
    # Unused mappings will cause KeyError if model doesn't have those weights
    return {
        # Common embeddings and output
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "lm_head.weight": "output.weight",
        
        # Attention weights (all models)
        "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
        "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
        "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
        "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
        
        # Attention bias weights (Qwen2 only)
        "model.layers.{}.self_attn.q_proj.bias": "layers.{}.attention.wq.bias",
        "model.layers.{}.self_attn.k_proj.bias": "layers.{}.attention.wk.bias",
        "model.layers.{}.self_attn.v_proj.bias": "layers.{}.attention.wv.bias",
        
        # QK normalization weights (Qwen3 only)
        "model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_norm.weight",
        "model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_norm.weight",
        
        # MLP weights (all models)
        'model.layers.{}.mlp.gate_proj.weight': 'layers.{}.feed_forward.w1.weight',
        "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
        "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
        
        # Normalization weights (all models)
        "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
        "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
        "model.norm.weight": "norm.weight",
        
        # Ignored weights
        'model.layers.{}.self_attn.rotary_emb.inv_freq': None,
    }


@torch.inference_mode()
def convert_hf_checkpoint(
    *,
    checkpoint_dir: Path = Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf"),
    model_name: Optional[str] = None,
) -> None:
    if model_name is None:
        model_name = checkpoint_dir.name

    # Detect model type and get appropriate configuration
    model_type = detect_model_type(model_name)
    config = get_model_config(model_name, model_type)
    weight_map = get_weight_mapping()  # Now unified for all model types
    
    print(f"Detected model type: {model_type}")
    print(f"Model config {config.__dict__}")

    # Load the json file containing weight mapping
    model_map_json_safetensors = checkpoint_dir / 'model.safetensors.index.json'
    model_map_json_pytorch = checkpoint_dir / "pytorch_model.bin.index.json"
    model_map_json = None
   
    try:
      assert model_map_json_safetensors.is_file()
      model_map_json = model_map_json_safetensors
      print(f"Found safetensors index at {model_map_json_safetensors}")
    except AssertionError:
      print(f"{model_map_json_safetensors} not found")
    if model_map_json is None:
      try:
        assert model_map_json_pytorch.is_file()
        model_map_json = model_map_json_pytorch
        print(f"Found pytorch index at {model_map_json_pytorch}")
      except AssertionError:
        print(f"{model_map_json_pytorch} not found")
   
    # if model_map_json is None: raise Exception("No model map found!")

    # with open(model_map_json) as json_map:
    #     bin_index = json.load(json_map)

    # CHANGE START: support single-file safetensors
    single_safetensors = checkpoint_dir / "model.safetensors"  # CHANGE
    if model_map_json is None and single_safetensors.is_file():  # CHANGE
        print(f"Found single safetensors file at {single_safetensors}")  # CHANGE
        temp_state = load_safetensors_file(str(single_safetensors), device="cpu")  # CHANGE
        bin_index = {"weight_map": {k: "model.safetensors" for k in temp_state.keys()}}  # CHANGE)
    else:  # CHANGE
        if model_map_json is None:  # CHANGE
            raise Exception("No model map found!")  # CHANGE
        with open(model_map_json) as json_map:
            bin_index = json.load(json_map)
    # CHANGE END

    bin_files = {checkpoint_dir / bin for bin in bin_index["weight_map"].values()}

    def permute(w, n_head):
        dim = config.dim
        print(f"dim: {dim}")
        print(f"Shape: {w.shape}")
        print(f"n_head: {n_head}")
        print(f"Head dim: {config.head_dim} ")
        return (
            w.view(n_head, 2, config.head_dim // 2, dim)
            .transpose(1, 2)
            .reshape(config.head_dim * n_head, dim)
        )

    merged_result = {}
    for file in sorted(bin_files):
       if "safetensors" in str(file):
           state_dict = load_safetensors_file(str(file), device="cpu")
           merged_result.update(state_dict)
       else:
           state_dict = torch.load(str(file), map_location="cpu", mmap=True, weights_only=True)
           merged_result.update(state_dict)
    final_result = {}

    # Llama 3.2 1b uses weight tied embedding https://huggingface.co/meta-llama/Llama-3.2-1B/discussions/99
    if not "lm_head.weight" in bin_index["weight_map"] and "llama-3.2-" in model_name.lower() :
        new_key = "output.weight"
        key = "model.embed_tokens.weight"
        final_result[new_key] = merged_result[key]


    for key, value in merged_result.items():
        if "layers" in key:
            abstract_key = re.sub(r'(\d+)', '{}', key)
            layer_num = re.search(r'\d+', key).group(0)
            new_key = weight_map[abstract_key]
            if new_key is None:
                continue
            new_key = new_key.format(layer_num)
        else:
            new_key = weight_map[key]

        final_result[new_key] = value

    for key in tuple(final_result.keys()):
        if "wq" in key:
            q = final_result[key]
            k = final_result[key.replace("wq", "wk")]
            v = final_result[key.replace("wq", "wv")]
            q = permute(q, config.n_head)
            k = permute(k, config.n_local_heads)
            final_result[key.replace("wq", "wqkv")] = torch.cat([q, k, v])
            del final_result[key]
            del final_result[key.replace("wq", "wk")]
            del final_result[key.replace("wq", "wv")]
    print(f"Saving checkpoint to {checkpoint_dir / 'model.pth'}")
    torch.save(final_result, checkpoint_dir / "model.pth")
    if 'llama-3' in model_name.lower():
        if 'llama-3.1-405b' in model_name.lower():
            original_dir = checkpoint_dir / "original" / "mp16"
        else:
            original_dir = checkpoint_dir / "original"
        tokenizer_model = original_dir / "tokenizer.model"
        tokenizer_model_tiktoken = checkpoint_dir / "tokenizer.model"
        print(f"Copying {tokenizer_model} to {tokenizer_model_tiktoken}")
        shutil.copy(tokenizer_model, tokenizer_model_tiktoken)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert HuggingFace checkpoint.')
    parser.add_argument('--checkpoint_dir', type=Path, default=Path("checkpoints/meta-llama/llama-2-7b-chat-hf"))
    parser.add_argument('--model_name', type=str, default=None)

    args = parser.parse_args()
    convert_hf_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model_name,
    )
