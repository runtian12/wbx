import os
os.environ["TORCH_LOGS"]="recompiles"
import itertools
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch._dynamo.config
import torch._inductor.config
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, create_block_mask
from torch._inductor import config as iconfig


import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
import math
import transformers
import torch
import time
import argparse
from colorama import Fore, Style
from dataclasses import dataclass
from torch import Tensor, nn
from copy import deepcopy
import builtins

# save the real print so rank 0 can still use it if you ever want to restore it
_real_print = builtins.print



def _get_model_size(model):
    model_size = 0
    params = 0
    for name, child in model.named_children():
        if not isinstance(child, torch.nn.Embedding):
            model_size += sum(
                [
                    p.numel() * p.dtype.itemsize
                    for p in itertools.chain(child.parameters(), child.buffers())
                ]
            )
            params += sum(
                [
                    p.numel()
                    for p in itertools.chain(child.parameters(), child.buffers())
                ]
            )
    return model_size, params


def multinomial_sample_one_no_sync(probs_sort): # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def apply_top_k_only(
    logits: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    pivot = v.select(-1, -1).unsqueeze(-1)
    logits = torch.where(logits < pivot, -float("Inf"), logits)
    print(torch.sum(~torch.isinf(logits), dim=-1))
    return logits

# Based on https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/topk_topp_sampler.py
def apply_top_k_top_p(
    logits: torch.Tensor,
    k: Optional[int],
    p: Optional[float],
) -> torch.Tensor:
    """Apply top-k and top-p masks to the logits.

    If a top-p is used, this function will sort the logits tensor,
    which can be slow for large batches.

    The logits tensor may be updated in-place.
    """
    if p is None:
        if k is None:
            return logits
        # Avoid sorting vocab for top-k only case.
        return apply_top_k_only(logits, k)
        
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if k is not None:
        # Apply top-k.
        top_k_mask = logits_sort.size(-1) - k
        # Get all the top_k values.
        # top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort[..., -k].unsqueeze(dim=-1)
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    if p is not None:
        # Apply top-p.
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
        top_p_mask = probs_sum <= 1 - p
        # at least one
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # Re-sort the probabilities.
    logits = logits_sort.scatter(dim=-1, index=logits_idx, src=logits_sort)
    return logits


# def sample_greedy(logits):
#     probs = torch.nn.functional.softmax(logits, dim=-1)
#     idx_next = torch.argmax(probs, dim=-1, keepdim=True)
#     return idx_next, probs

def logits_to_probs(logits: torch.Tensor, temperature: float = 1.0, top_k: Optional[int] = None, top_p: Optional[float] = None):
    if temperature > 0:
        logits = logits.div_(max(temperature, 1e-5))
        logits = apply_top_k_top_p(logits, top_k, top_p)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None, top_p: Optional[float] = 0.95):
    probs = logits_to_probs(logits[:, -1], temperature, top_k, top_p)
    if temperature == 0:
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)
    else:
        idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs

def causal_mask(b, h, q, kv):
    return q >= kv

def roundup(val, multiplier):
    return ((val - 1) // multiplier + 1) * multiplier




compiled_decode_one_token_draft = None
compiled_decode_one_token_target = None
compiled_model_forward_target = None
compiled_model_forward_draft = None
terminators = None
default_device = None
prefill_target = None
prefill_draft = None
g_block_mask = None

# Compile create_block_mask for better performance
create_block_mask = torch.compile(create_block_mask)



def forwad_with_logits_and_embeddings_output(self, mask: BlockMask, idx: Tensor, input_pos: Optional[Tensor] = None, layer_index: int = None) -> Tensor:
    assert self.freqs_cis is not None, "Caches must be initialized first"
    mask.mask_mod = self.get_mask_mod(mask.mask_mod, input_pos[0])
    freqs_cis = self.freqs_cis[input_pos]
    x = self.tok_embeddings(idx)
    
    # Calculate target layer index (handles negative indexing)
    if layer_index is not None:
        num_layers = len(self.layers)
        target_layer_idx = (layer_index + num_layers) % num_layers
    else:
        target_layer_idx = None
    
    # Process layers and capture embedding at target layer
    target_embedding = None
    for i, layer in enumerate(self.layers):
        x = layer(x, input_pos, freqs_cis, mask)
        if i == target_layer_idx:
            target_embedding = x.clone()
    
    x = self.norm(x)
    logits = self.output(x)
    
    return logits, target_embedding

    
def patch_model(model_cls):
    model_cls.forward = forwad_with_logits_and_embeddings_output
    return model_cls


def detect_model_type(model_name: str) -> str:
    """Detect if this is a Qwen model based on directory name"""
    model_name_lower = model_name.lower()
    if "qwen3" in model_name_lower:
        return "qwen3"
    elif "qwen2" in model_name_lower or "qwen" in model_name_lower:
        return "qwen2"
    else:
        return "llama"

def get_model_cls(model_type:str):
    if model_type in ["qwen2", "qwen3"]:
        from .model_qwen import QwenTransformer
        model_cls = QwenTransformer
    elif model_type == "llama":
        from .model import Transformer
        model_cls = Transformer
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model_cls

def load_model_from_name(model_name: str):
    """Load appropriate model class based on model type"""
    model_type = detect_model_type(model_name)
    print(f"Detected model type: {model_type} for model name: {model_name}")
    model_cls = get_model_cls(model_type)
    patched_model_cls = patch_model(model_cls)
    print(f"Patched model class: {patched_model_cls}")
    print(f"Forwad: patched_model_cls.foward: {patched_model_cls.forward}")

    return patched_model_cls.from_name(model_name)



def setup_gpt_fast(target_model_name=None, draft_model_name=None, tokenizer=None, compile=True, _terminators=None, max_seq_len=None):
    global print
    # support running without installing as a package
    wd = Path(__file__).parent.parent.resolve()
    sys.path.append(str(wd))

    draft_checkpoint_path = Path(f"checkpoints/{draft_model_name}/model.pth")
    checkpoint_path = Path(f"checkpoints/{target_model_name}/model.pth")
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.triton.unique_kernel_names = True
    # Experimental features to reduce compilation times, will be on by default in future
    torch._inductor.config.fx_graph_cache = True 
    # torch._functorch.config.enable_autograd_cache = True

    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    iconfig.fallback_random = True
    # torch.use_deterministic_algorithms(True)



    assert checkpoint_path.is_file(), checkpoint_path

    # global print
    from src.gpt_fast.tp import maybe_init_dist
    rank = maybe_init_dist()
    use_tp = rank is not None
    if use_tp:
        if rank != 0:
            # only print on rank 0
            # print = lambda *args, **kwargs: None
            builtins.print = lambda *args, **kwargs: None

    print(f"Number of available GPUs: {torch.cuda.device_count()}")

    global default_device
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = default_device

    print(f"Using device={device}")
    precision = torch.bfloat16

    print("Loading model ...")
    t0 = time.time()

    target_model = _load_model(checkpoint_path, device, precision, use_tp) if target_model_name is not None else None

    print(f"Not using tensor parallelism, for the draft model")
    draft_model = _load_model(draft_checkpoint_path, device, precision, use_tp) if draft_model_name is not None else None

    device_sync(device=device) # MKG
    print(f"Time to load model: {time.time() - t0:.02f} seconds")


    from transformers import AutoTokenizer

    torch.manual_seed(1234)
    model_size, params = _get_model_size(target_model)

    if compile and use_tp:
        torch._inductor.config.triton.cudagraph_trees = False # Bug with cudagraph trees in this case

    global terminators
    if _terminators is None:
        terminators = [
            tokenizer.eos_token_id,
            # tokenizer.encode("<|end_of_text|>", add_special_tokens=False)[0]
        ]
        if "llama" in draft_model_name.lower():
            terminators.append(tokenizer.encode("<|end_of_text|>", add_special_tokens=False)[0])
    else:
        terminators = _terminators

    terminators = torch.tensor(terminators, device=device)



    global compiled_decode_one_token_draft
    global compiled_decode_one_token_target
    global compiled_model_forward_target
    global compiled_model_forward_draft
    global prefill_target, prefill_draft
    global g_block_mask
    if compile:
        if use_tp: # and ("cuda" in device):
            torch._inductor.config.triton.cudagraph_trees = False # Bug with cudagraph trees in this case



        compiled_decode_one_token_target = torch.compile(decode_one_token_target, mode="reduce-overhead", fullgraph=True)
        compiled_decode_one_token_draft = torch.compile(decode_one_token_draft, mode="reduce-overhead", fullgraph=True)
        compiled_model_forward_target = torch.compile(model_forward_target, mode="reduce-overhead", fullgraph=True)
        compiled_model_forward_draft = torch.compile(model_forward_draft, mode="reduce-overhead", fullgraph=True)
        # prefill_target = torch.compile(model_forward_target, mode="reduce-overhead", fullgraph=True, dynamic=True)
        # prefill_draft = torch.compile(model_forward_draft, mode="reduce-overhead", fullgraph=True, dynamic=True)
    else:
        compiled_decode_one_token_target = decode_one_token_target
        compiled_decode_one_token_draft = decode_one_token_draft
        compiled_model_forward_target = model_forward_target
        compiled_model_forward_draft = model_forward_draft
    prefill_target = model_forward_target
    prefill_draft = model_forward_draft

    with torch.device(device):
        target_model.setup_caches(max_batch_size=1, max_seq_length=max_seq_len)
        draft_model.setup_caches(max_batch_size=1, max_seq_length=max_seq_len)

    print(target_model.max_seq_length)
    g_block_mask = create_block_mask(causal_mask, 1, 1, target_model.max_seq_length, target_model.max_seq_length, device=device)
    print(f"Created block mask")
    return target_model, draft_model, rank, device


def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet suppported")




def _load_model(checkpoint_path, device, precision, use_tp):
    with torch.device('meta'):
        model = load_model_from_name(checkpoint_path.parent.name)

    if "int8" in str(checkpoint_path):
        print("Using int8 weight-only quantization!")
        from .quantize import WeightOnlyInt8QuantHandler
        simple_quantizer = WeightOnlyInt8QuantHandler(model)
        model = simple_quantizer.convert_for_runtime()

    if "int4" in str(checkpoint_path):
        print("Using int4 weight-only quantization!")
        path_comps = checkpoint_path.name.split(".")
        groupsize = int(path_comps[-2][1:])
        from .quantize import WeightOnlyInt4QuantHandler
        simple_quantizer = WeightOnlyInt4QuantHandler(model, groupsize)
        model = simple_quantizer.convert_for_runtime()

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, assign=True)

    if use_tp:
        from src.gpt_fast.tp import apply_tp
        print("Applying tensor parallel to model ...")
        apply_tp(model)

    model = model.to(device=device, dtype=precision)
    return model.eval()

def model_forward_target(model, x, input_pos, block_mask, layer_index=-1):
    block_index = input_pos // block_mask.BLOCK_SIZE[0]
    mask = block_mask[:, :, block_index]
    mask.mask_mod = block_mask.mask_mod
    mask.seq_lengths = (input_pos.shape[0], model.max_seq_length)
    # return model(mask, x, input_pos)
    logits, embeddings = model(mask, x, input_pos, layer_index=layer_index)
    return logits, embeddings

def model_forward_draft(model, x, input_pos, block_mask):
    block_index = input_pos // block_mask.BLOCK_SIZE[0]
    mask = block_mask[:, :, block_index]
    mask.mask_mod = block_mask.mask_mod
    mask.seq_lengths = (input_pos.shape[0], model.max_seq_length)
    # return model(mask, x, input_pos)
    logits, embeddings = model(mask, x, input_pos)
    return logits, embeddings



def decode_one_token_target(model, x: torch.Tensor, input_pos: torch.Tensor, block_mask: BlockMask, **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    block_index = input_pos // block_mask.BLOCK_SIZE[0]
    mask = block_mask[:, :, block_index]
    mask.mask_mod = block_mask.mask_mod
    mask.seq_lengths = (1, model.max_seq_length)
    logits, _ = model(mask, x, input_pos)
    return sample(logits, **sampling_kwargs)

def decode_one_token_draft(model, x: torch.Tensor, input_pos: torch.Tensor, block_mask: BlockMask, **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    block_index = input_pos // block_mask.BLOCK_SIZE[0]
    mask = block_mask[:, :, block_index]
    mask.mask_mod = block_mask.mask_mod
    mask.seq_lengths = (1, model.max_seq_length)
    logits, _ = model(mask, x, input_pos)
    return sample(logits, **sampling_kwargs)


def decode_n_tokens(model, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, callback=lambda _: _, is_target=True, return_probs=True, **sampling_kwargs):
    if is_target:
        compiled_decode_one_token = compiled_decode_one_token_target
        model_forward_func = compiled_model_forward_target
        if input_pos.shape[-1] > 50:
            model_forward_func = prefill_target
            print("Using prefill for model forward")
    else:
        compiled_decode_one_token = compiled_decode_one_token_draft
        model_forward_func = compiled_model_forward_draft
        if input_pos.shape[-1] > 50:
            model_forward_func = prefill_draft
            print("Using prefill for model forward")

    # Create block mask once for the entire sequence generation
    block_mask = g_block_mask

    new_tokens, new_probs = [], []
    if input_pos.shape[-1] != 1:
        t = time.time()
        logits, _ = model_forward_func(model, cur_token, input_pos, block_mask)
        # print(f"Time to get logits for input_pos {input_pos.shape} {'target' if is_target else 'draft'}: {time.time() - t:.04f} seconds")
        next_token, next_prob = sample(logits, **sampling_kwargs)
        new_tokens.append(next_token.clone())
        callback(new_tokens[-1])
        if return_probs:
            new_probs.append(next_prob.clone())
        cur_token = next_token.clone()
        input_pos = torch.tensor([input_pos[-1] + 1], device=cur_token.device)
        num_new_tokens -= 1
    start = time.perf_counter()
    for i in range(num_new_tokens):
        iter_start = time.perf_counter()
        # if cur_token[0, 0] in terminators:
        #     break
        # t = time.perf_counter()
        if (cur_token[0, 0] == terminators).any(): # More efficient way to check if in terminators
            break
        # print(f"Time to check for terminators: {time.perf_counter() - t:.04f} seconds")
        cur_token = cur_token.contiguous()
        t = time.time()
        next_token, next_prob = compiled_decode_one_token(
            model, cur_token, input_pos, block_mask, **sampling_kwargs
        )
        # print(f"{i} Time to decode one token for input_pos {input_pos.shape} {'target' if is_target else 'draft'}: {time.time() - t:.04f} seconds")
        input_pos += 1
        new_tokens.append(next_token.clone())
        callback(new_tokens[-1])
        if return_probs:
            new_probs.append(next_prob.clone())
        cur_token = next_token.clone()

    return new_tokens, new_probs


@torch.no_grad()
def complete_using_generation(
    model,
    tokenizer,
    prefix,
    max_new_tokens,
    max_seq_len,
    is_target,
    past_key_values, # number of cached tokens
    start_index=0,
    do_sample=False,
    num_beams=1,
    num_return_sequences=1,
    return_dict_in_generate=True,
    fixed_length=1,
    temperature=0,
    top_k=10,
    top_p=None,
    return_probs=True
):
    if past_key_values is None:
        past_key_values = 0
    past_key_values_idx = past_key_values - 1
    assert prefix.shape[0] == 1, "Batch size should be 1"
    input_ids = prefix
    assert type(input_ids) == torch.Tensor

    max_new_tokens = min(
        max_new_tokens, max_seq_len - input_ids.shape[1] - 1
    )  # -1 for the last token
    input_ids = input_ids.to(default_device)
    original_input_len = input_ids.shape[1]

    n_not_cached_tokens = original_input_len - past_key_values

    if n_not_cached_tokens > 1 and n_not_cached_tokens < fixed_length:
        not_cached_tokens = input_ids[:, -fixed_length:].clone(memory_format=torch.contiguous_format)
        input_pos = torch.arange(original_input_len - fixed_length, original_input_len, device=input_ids.device)
    else:
        not_cached_tokens = input_ids[:, past_key_values_idx+1:].clone(memory_format=torch.contiguous_format)
        input_pos = torch.arange(past_key_values_idx+1, original_input_len, device=input_ids.device)
    new_tokens, new_probs = decode_n_tokens(model, not_cached_tokens, input_pos, max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p, is_target=is_target)

    if return_probs:
        new_tokens = torch.cat(new_tokens).view(1, -1) if len(new_tokens) > 0 else torch.empty((1, 0), device=input_ids.device, dtype=input_ids.dtype)
        new_probs = torch.cat(new_probs).unsqueeze(0) if len(new_probs) > 0 else torch.empty((1, 0), device=input_ids.device, dtype=input_ids.dtype)
    else:
        new_tokens = torch.cat(new_tokens).view(1, -1) if len(new_tokens) > 0 else torch.empty((1, 0), device=input_ids.device, dtype=input_ids.dtype)
        new_probs = None

    sequence = torch.cat([input_ids, new_tokens], dim=1)
    pask_key_values = sequence.shape[1] - 1
    return sequence, new_probs, pask_key_values


def prepare_forward_inputs(prefix, past_key_values, max_seq_len, fixed_length):
    if past_key_values is None:
        past_key_values = 0
    past_key_values_idx = past_key_values - 1
    assert prefix.shape[0] == 1, "Batch size should be 1"
    input_ids = prefix
    assert type(input_ids) == torch.Tensor
    input_ids = input_ids.to(default_device)
    original_input_len = input_ids.shape[1]
    input_pos = torch.arange(past_key_values_idx+1, original_input_len, device=input_ids.device)
    not_cached_tokens = input_ids[:, past_key_values_idx+1:]
    return not_cached_tokens.clone(memory_format=torch.contiguous_format), input_pos.clone(memory_format=torch.contiguous_format)



@torch.no_grad()
def get_logits_and_emneddings(model, prefix, past_key_values, is_target=True, max_seq_len=512, fixed_length=21, layer_index=-1):
    if past_key_values is None:
        past_key_values = 0
    not_cached_tokens, input_pos = prepare_forward_inputs(prefix, past_key_values, max_seq_len, fixed_length)
    block_mask = g_block_mask
    if is_target:
        model_forward_func = compiled_model_forward_target
        if input_pos.shape[-1] > 50:
            model_forward_func = prefill_target
            print("Using prefill for model forward")
    else:
        model_forward_func = compiled_model_forward_draft
        if input_pos.shape[-1] > 50:
            model_forward_func = prefill_draft
            print("Using prefill for model forward")
    # Create prefill mask based on actual sequence length
    t = time.time()
    logits, embeddings = model_forward_func(model, not_cached_tokens, input_pos, block_mask=block_mask ,layer_index=layer_index)
    assert len(embeddings.shape) == 3, f"Embeddings shape: {embeddings.shape}"
    assert len(logits.shape) == 3, f"Logits shape: {logits.shape}"
    assert len(prefix.shape) == 2, f"Prefix shape: {prefix.shape}"
    past_key_values = prefix.shape[1] + logits.shape[1] - 1
    return logits, embeddings, past_key_values
