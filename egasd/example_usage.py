"""Command-line EGASD example.

Run from the repository root:

    python -m egasd.example_usage --device cpu --max_new_tokens 16 --no_pivot
"""

from __future__ import annotations

import argparse
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .egasd_decode import EGASDConfig, EGASDDecoder
    from .models import AcceptancePredictionHead, PivotClassifier
except ImportError:
    from egasd_decode import EGASDConfig, EGASDDecoder
    from models import AcceptancePredictionHead, PivotClassifier


DEFAULT_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def resolve_device(requested_device: str) -> str:
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.")
        return "cpu"
    return requested_device


def load_model(model_name: str, device: str, bf16: bool):
    dtype = torch.bfloat16 if bf16 and device.startswith("cuda") else torch.float32
    if device.startswith("cuda"):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(device)
    model.eval()
    return model


def build_optional_heads(args, draft_model, target_model, device: str):
    acceptance_head = None
    pivot_classifier = None

    if args.use_random_acceptance_head:
        acceptance_head = AcceptancePredictionHead(
            {
                "hidden_size": draft_model.config.hidden_size,
                "num_layers": args.num_res_layers,
            }
        ).to(device)
        acceptance_head.eval()

    if not args.no_pivot:
        pivot_classifier = PivotClassifier(
            target_hidden_dim=target_model.config.hidden_size,
        ).to(device)
        pivot_classifier.eval()

    return acceptance_head, pivot_classifier


def run_generation(args, prompts: Iterable[str]) -> None:
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"Draft model: {args.draft_model}")
    print(f"Target model: {args.target_model}")

    tokenizer = AutoTokenizer.from_pretrained(args.draft_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading draft model...")
    draft_model = load_model(args.draft_model, device, args.bf16)

    print("Loading target model...")
    target_model = load_model(args.target_model, device, args.bf16)

    acceptance_head, pivot_classifier = build_optional_heads(
        args,
        draft_model,
        target_model,
        device,
    )

    config = EGASDConfig(
        h_min=args.h_min,
        h_max=args.h_max,
        min_draft_length=args.min_draft_length,
        max_draft_length=args.max_draft_length,
        pivot_threshold=args.pivot_threshold,
        relaxation_factor=args.relaxation_factor,
        do_sample=args.temperature > 0,
        temperature=max(args.temperature, 1e-5),
        top_k=args.top_k,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        use_pivot_verification=not args.no_pivot,
    )

    decoder = EGASDDecoder(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        acceptance_head=acceptance_head,
        pivot_classifier=pivot_classifier,
        config=config,
        device=device,
    )

    for prompt in prompts:
        print("\n" + "=" * 80)
        print(f"Prompt: {prompt}")

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)

        decoder.reset_stats()
        generated_ids, stats = decoder.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
        )

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"Output: {generated_text}")
        print("Stats:")
        print(f"  total_tokens: {stats['total_tokens']}")
        print(f"  accepted_tokens: {stats['accepted_tokens']}")
        print(f"  draft_iterations: {stats['draft_iterations']}")
        print(f"  total_draft_tokens: {stats['total_draft_tokens']}")
        print(f"  acceptance_rate: {stats['acceptance_rate']:.2%}")
        print(f"  avg_entropy: {stats.get('avg_entropy', 0):.4f}")
        print(f"  avg_threshold: {stats.get('avg_threshold', 0):.4f}")
        print(f"  inference_time: {stats['inference_time']:.2f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an EGASD decoding example.")
    parser.add_argument("--draft_model", default=DEFAULT_MODEL)
    parser.add_argument("--target_model", default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--min_draft_length", type=int, default=2)
    parser.add_argument("--max_draft_length", type=int, default=20)
    parser.add_argument("--h_min", type=float, default=0.4)
    parser.add_argument("--h_max", type=float, default=0.8)
    parser.add_argument("--pivot_threshold", type=float, default=0.5)
    parser.add_argument("--relaxation_factor", type=float, default=1.2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--no_pivot", action="store_true")
    parser.add_argument("--use_random_acceptance_head", action="store_true")
    parser.add_argument("--num_res_layers", type=int, default=1)
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=[
            "Artificial intelligence is",
            "Machine learning means",
        ],
        help="One or more prompts to decode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_generation(args, args.prompts)


if __name__ == "__main__":
    main()
