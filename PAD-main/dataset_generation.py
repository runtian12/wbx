import os
import time
import json
import argparse

import torch
import numpy as np
import transformers
from dotenv import load_dotenv
from colorama import Fore, Style
from vllm import LLM, SamplingParams

from src.util.json_io import save_line_jsonlines
from src.util.gsm8k_helper import *
from src.util.tasks.mbpp import generate_prompt_mbpp
from src.util.general import get_text
from src.util.tasks import TaskManager

# Load environment variables from the .env file (if present)
load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

# Set random seeds for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def log_memory_usage(label=""):
    """Log current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"[{label}] GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    else:
        print(f"[{label}] CUDA not available")


rank = None

def complete_using_generation(
    model,
    tokenizer,
    prefix,
    max_new_tokens,
    max_seq_len,
    is_target,
    past_key_values=None,
    start_index=0,
    do_sample=False,
    num_beams=1,
    num_return_sequences=1,
    return_dict_in_generate=True
):
    """
    Generate text using VLLM model.
    
    Note: VLLM manages KV cache internally based on prompt_token_ids, 
    unlike HuggingFace models which accept past_key_values explicitly.
    """
    max_new_tokens = max(max_new_tokens, 0)
    input_ids_tensor = prefix
    assert type(input_ids_tensor) == torch.Tensor

    original_input_len = input_ids_tensor.shape[1]

    # Convert input_ids_tensor to list of token IDs for VLLM
    input_token_ids_list = input_ids_tensor.tolist()

    sampling_params = SamplingParams(
        n=num_return_sequences,
        temperature=0.0 if not do_sample else 0.7,
        max_tokens=max_new_tokens,
        stop_token_ids=stop_token_ids,
    )

    start = time.time()
    vllm_outputs = model.generate(prompt_token_ids=input_token_ids_list, sampling_params=sampling_params, use_tqdm=False)
    end_time = time.time()

    print(f"VLLM Time taken: {end_time - start}, Input length: {original_input_len}, Num output seqs: {len(vllm_outputs[0].outputs)}, Model: VLLM ({target_model_name})")

    # Process VLLM outputs (list of RequestOutput objects)
    sequences_list = []
    for completion_output in vllm_outputs[0].outputs:
        sequences_list.append(input_token_ids_list[0] + completion_output.token_ids)

    # Pad sequences to the same length to create a tensor
    max_len = max(len(seq) for seq in sequences_list)
    padded_sequences = []
    for seq in sequences_list:
        padding_needed = max_len - len(seq)
        padded_seq = seq + [pad_token_id] * padding_needed 
        padded_sequences.append(padded_seq)
    
    sequences_tensor = torch.tensor(padded_sequences, dtype=torch.long).to(input_ids_tensor.device)
    
    # VLLM returns logprobs, not full logits
    scores_tensor = None
    if return_dict_in_generate and vllm_outputs[0].outputs[0].logprobs:
        raise NotImplementedError()

    print(f"VLLM Sequence shape: {sequences_tensor.shape}")
    return sequences_tensor, scores_tensor, None


def select_kv_cache_portion(past_key_values, start_idx, end_idx):
    """Placeholder for KV cache selection (not used with VLLM)"""
    return None

def get_dataset(dataset):
    base_paths = {
        "gsm8k": "data/gsm8k",
        "math": "data/math_splits",
    }
    base_path = base_paths.get(dataset, None)
    return TaskManager.load_dataset(dataset, base_path=base_path)


def get_utility_rate_by_dataset(
    sequences_or_texts,
    reference_sequence_or_text,
    init_len,
    tokenizer,
    dataset=None,
    sample=None,
    record=None,
    target_model=None,
    self_supervised=False,
    max_extra_len=50,
    prev_len=0,
    self_correction_check_mode=False,
    max_extra_len_ratio=None,
    mlc=None,
    **kwargs
):
    """Calculate the utility rate (correctness ratio) for a set of sequences"""
    if sequences_or_texts is None:
        return 0.0

    if reference_sequence_or_text:
        reference_txt = get_text(reference_sequence_or_text, tokenizer, skip_special_tokens=True)
        polished_reference = polish_ground_truth_by_dataset(reference_txt, tokenizer, dataset=dataset, sample=sample, **kwargs)
    else:
        polished_reference = None

    seq_correctness = [
        check_correctness_by_dataset(
            sequence, polished_reference, init_len=init_len, tokenizer=tokenizer, dataset=dataset, polish_reference=False, sample=sample, **kwargs
        )
        for sequence in sequences_or_texts
    ]

    num_valid = sum(seq_correctness)
    ratio = num_valid / len(sequences_or_texts)
    return ratio


def polish_ground_truth_by_dataset(reference_txt, tokenizer, start_idx=0, dataset=None, sample=None):
    """Wrapper for TaskManager.polish_ground_truth"""
    return TaskManager.polish_ground_truth(dataset, reference_txt)


def check_correctness_by_dataset(
    sequence_or_text, polished_reference_txt, init_len=0, tokenizer=None, dataset=None, sample=None, **kwargs
):
    """Wrapper for TaskManager.evaluate"""
    return TaskManager.evaluate(
        dataset,
        sequence_or_text,
        polished_reference_txt,
        sample=sample,
        start_idx=init_len,
        tokenizer=tokenizer,
        polish_reference=False,
    )


def utility_changed_sampling(
    model_target,
    model_draft,
    tokenizer,
    dataset,
    prefix_accepted,
    prefix_rejected,
    max_new_tokens,
    max_seq_len,
    init_len,
    prev_len,
    num_samples=20,
    self_supervised=True,
    sampling_rate_threshold=0.9,
    original_ground_truth="",
    num_beams=1,
    lottery_token=False,
    target_past_key_values_rejected=None,
    target_past_key_values_accepted=None,
    sample=None,
    max_extra_len_ratio=None,
    self_correction_check_mode=None,
    mlc=None,
    record=None,
):
    """
    Check if utility has changed by sampling from accepted and rejected prefixes.
    Returns whether utility dropped and generated sequences.
    """
    input_ids_accepted = prefix_accepted
    input_ids_rejected = prefix_rejected
    assert type(input_ids_accepted) == torch.Tensor
    assert type(input_ids_rejected) == torch.Tensor

    do_sample = num_samples > 1
    return_dict_in_generate = args.kv_caching

    with torch.no_grad():
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        # Generate continuations for rejected sequence
        if not any(stop_token in input_ids_rejected[0][init_len:] for stop_token in stop_token_ids):
            if args.follow_target_trajectory:
                raise NotImplementedError("Following target trajectory is not implemented")
            
            outputs_rejected, _, _ = complete_using_generation(
                model_target,
                tokenizer,
                input_ids_rejected,
                max_new_tokens,
                max_seq_len=max_seq_len,
                is_target=True,
                past_key_values=target_past_key_values_rejected,
                num_beams=1,
                num_return_sequences=num_samples,
                return_dict_in_generate=return_dict_in_generate,
                do_sample=do_sample,
            )
        else:
            print("Rejected Sequence already reached EOS")
            print(f"Rejected Sequence: {tokenizer.decode(input_ids_rejected[0][prev_len:], skip_special_tokens=True)}")
            outputs_rejected = input_ids_rejected

        print(f"Len of outputs_rejected: {len(outputs_rejected)}")
        print(f"Ouputs_rejected: {outputs_rejected.shape}")

        # Generate continuations for accepted sequence
        if not any(stop_token in input_ids_accepted[0][init_len:] for stop_token in stop_token_ids):
            outputs_accepted, _, past_key_values_accepted = complete_using_generation(
                model_target,
                tokenizer,
                input_ids_accepted,
                max_new_tokens,
                max_seq_len=max_seq_len,
                is_target=True,
                past_key_values=target_past_key_values_accepted,
                do_sample=do_sample,
                num_beams=1,
                num_return_sequences=num_samples,
                return_dict_in_generate=return_dict_in_generate,
            )
        else:
            print("Accepted Sequence already reached EOS")
            print(f"Accepted Sequence: {tokenizer.decode(input_ids_accepted[0][prev_len:], skip_special_tokens=True)}")
            outputs_accepted = input_ids_accepted
            past_key_values_accepted = target_past_key_values_accepted if args.kv_caching else None

        print(f"Ouputs_accepted: {outputs_accepted.shape}")

    sequences_accepted = outputs_accepted
    sequences_rejected = outputs_rejected

    prob_accepted = get_utility_rate_by_dataset(
        sequences_accepted,
        original_ground_truth,
        init_len,
        tokenizer,
        dataset=dataset,
        sample=sample,
        record=record,
        target_model=target_model,
        self_supervised=self_supervised,
        max_extra_len=max_extra_len_ratio,
        prev_len=prev_len,
        self_correction_check_mode=self_correction_check_mode,
        mlc=mlc,
    )

    prob_rejected = get_utility_rate_by_dataset(
        sequences_rejected,
        original_ground_truth,
        init_len,
        tokenizer,
        dataset=dataset,
        sample=sample,
        record=record,
        target_model=target_model,
        self_supervised=self_supervised,
        max_extra_len=max_extra_len_ratio,
        prev_len=prev_len,
        self_correction_check_mode=None,  # No need to check self-correction for rejected sequences
        mlc=mlc,
    )

    print(f"Target Probability of accepted: {prob_accepted}, Target Probability of rejected: {prob_rejected}")
    print(f"Target Utility changed: {prob_accepted != prob_rejected}")
    record["prob_accepted"] = prob_accepted
    record["prob_rejected"] = prob_rejected

    # Determine if utility dropped based on comparison type
    if args.ratio_compare_type == "mul":
        utility_dropped = prob_accepted < prob_rejected * sampling_rate_threshold
    elif args.ratio_compare_type == "add":
        utility_dropped = prob_accepted + sampling_rate_threshold < prob_rejected
    elif args.ratio_compare_type == "sa":
        utility_dropped = prob_accepted < sampling_rate_threshold
    else:
        utility_dropped = False

    print("Utility dropped significantly" if utility_dropped else "Utility did not drop significantly")

    return utility_dropped, outputs_accepted, outputs_rejected, past_key_values_accepted


def strike(text):
    """Apply strikethrough formatting to text"""
    return "".join(["{}\u0336".format(c) for c in text])


def validate_token(candidate_input_ids, target_input_ids, candidate_logits, target_logits):
    """
    Validate if candidate tokens match target tokens.
    Returns: (all_valid, n_verified_tokens, directly_pivotal)
    """
    all_valid = True
    n_verified_tokens = 0
    directly_pivotal = False
    
    for i in range(len(candidate_input_ids)):
        candidate_token_id = candidate_input_ids[i].item()
        target_token_id = target_input_ids[i].item()
        if candidate_token_id != target_token_id:
            all_valid = False
            directly_pivotal = True
            print(f"Candidate token: {tokenizer.decode([candidate_token_id])}, Target token: {tokenizer.decode([target_token_id])}")
            break
        n_verified_tokens += 1

    return all_valid, n_verified_tokens, directly_pivotal


def generate_label(tokens, n_new_tokens, n_verified_tokens, pivotal):
    """
    Generate labels for tokens: 0 for previous, -1 for verified, pivotal for rejected.
    """
    prev_len = len(tokens) - n_new_tokens
    rejected_len = n_new_tokens - n_verified_tokens
    labels = (
        [0 for _ in range(prev_len)]
        + [-1 for _ in range(n_verified_tokens)]
        + [pivotal for _ in range(rejected_len)]
    )
    assert len(labels) == len(tokens), f"Labels length {len(labels)} does not match tokens length {len(tokens)}"
    return labels


def get_shortest_with_correct_answer(
    sequences, reference_sequence_or_text, start_idx, tokenizer, dataset=None, sample=None
):
    """
    Find the sequence with median length that contains the correct answer.
    Returns the median correct sequence and list of all correct answer lengths.
    """
    if reference_sequence_or_text:
        reference_txt = get_text(reference_sequence_or_text, tokenizer, skip_special_tokens=True)
        polished_reference_txt = polish_ground_truth_by_dataset(reference_txt, tokenizer, dataset=dataset, sample=sample)
    else:
        polished_reference_txt = None

    correct_sequences = []
    len_correct_answers = []
    pad_token_id = tokenizer.pad_token_id
    
    for sequence in sequences:
        if check_correctness_by_dataset(sequence, polished_reference_txt, init_len=start_idx, tokenizer=tokenizer, dataset=dataset, sample=sample):
            # Sequence length is until end or the first pad token
            seq_len = (sequence != pad_token_id).sum(-1)
            len_correct_answers.append(seq_len.item())
            assert sequence.dim() == 1, "Sequence should be a 1D tensor"
            correct_sequences.append((sequence, seq_len.item()))

    if not correct_sequences:
        return None, len_correct_answers
    
    # Sort by length and find median
    correct_sequences.sort(key=lambda x: x[1])
    median_idx = len(correct_sequences) // 2
    median_sequence = correct_sequences[median_idx][0]

    return median_sequence, len_correct_answers

def prepare_sample_by_dataset(train_data, data_index, tokenizer, dataset=None):
    """Prepare a sample based on the dataset type. Wrapper for TaskManager.prepare_sample"""
    return TaskManager.prepare_sample(dataset, train_data, data_index, tokenizer)


def generate_dataset(
    target_model,
    draft_model,
    tokenizer,
    data_index,
    train_data,
    max_iter=100,
    spec_len=1,
    output_dataset_dir=None,
    self_supervised=True,
    sampling_num=1,
    sampling_rate_threshold=0.9,
    num_beams=1,
    max_seq_len=None,
    max_extra_len=None,
    max_extra_len_ratio=None,
    self_correction_check_mode=None,
    mlc=None,
    log_dir=None
):
    """
    Generate training dataset by iteratively comparing draft and target model outputs.
    Identifies pivotal tokens where draft model diverges from target model in utility.
    """
    current_sequence, ground_truth, prompt, input_sample = prepare_sample_by_dataset(
        train_data, data_index, tokenizer, dataset=args.dataset
    )

    start_time = time.time()
    init_len = current_sequence.shape[1]
    log = {"prompt": prompt, "ground_truth": ground_truth}
    steps = []
    formatted_output = ""

    # Generate initial completions from target and draft models
    completion_using_target, _, _ = complete_using_generation(
        target_model,
        tokenizer,
        current_sequence,
        max_seq_len=max_seq_len,
        max_new_tokens=max_seq_len - init_len,
        is_target=True,
        past_key_values=None,
        do_sample=True,
        num_return_sequences=args.sampling_num,
    )
    
    if args.follow_target_trajectory:
        # Generate greedy path to follow during generation
        greedy_completion_using_target, _, _ = complete_using_generation(
            target_model,
            tokenizer,
            current_sequence,
            max_seq_len=max_seq_len,
            max_new_tokens=max_seq_len - init_len,
            is_target=True,
            past_key_values=None,
        )

    completion_using_draft, _, _ = complete_using_generation(
        draft_model,
        tokenizer,
        current_sequence,
        max_seq_len=max_seq_len,
        max_new_tokens=max_seq_len - init_len,
        is_target=True,
        past_key_values=None,
        do_sample=False,
        num_return_sequences=1,
    )

    # Evaluate initial responses
    max_seq_len = min(max_seq_len, completion_using_target.shape[1] + max_extra_len)
    
    try:
        target_response_utility_rate = get_utility_rate_by_dataset(
            completion_using_target,
            ground_truth,
            init_len,
            tokenizer,
            dataset=args.dataset,
            sample=input_sample,
            record=None,
            target_model=target_model,
            self_supervised=self_supervised,
            max_extra_len=max_extra_len_ratio,
            self_correction_check_mode=self_correction_check_mode,
            mlc=mlc
        )
    except Exception as e:
        log["error"] = str(e)
        return ("", [], log)
    draft_response_utility_rate = get_utility_rate_by_dataset(
        completion_using_draft,
        ground_truth,
        init_len,
        tokenizer,
        dataset=args.dataset,
        sample=input_sample,
        record=None,
        target_model=target_model,
        self_supervised=self_supervised,
        max_extra_len=max_extra_len_ratio,
        self_correction_check_mode=self_correction_check_mode,
        mlc=mlc
    )

    # Get shortest correct answers from target and draft
    target_shortest_answer, _ = get_shortest_with_correct_answer(
        completion_using_target,
        ground_truth,
        init_len,
        tokenizer,
        dataset=args.dataset,
        sample=input_sample,
    )
    
    draft_shortest_answer, _ = get_shortest_with_correct_answer(
        completion_using_draft,
        ground_truth,
        init_len,
        tokenizer,
        dataset=args.dataset,
        sample=input_sample,
    )

    target_shortest_correct_txt = tokenizer.decode(
        target_shortest_answer[init_len:] if target_shortest_answer is not None else completion_using_target[0][init_len:],
        skip_special_tokens=True
    )
    draft_shortest_correct_txt = tokenizer.decode(
        draft_shortest_answer[init_len:] if draft_shortest_answer is not None else completion_using_draft[0][init_len:],
        skip_special_tokens=True
    )
    # Check soundness of target's shortest answer
    target_shortest_soundness = None
    target_soundness_check_failed = False
    target_shortest_soundness_analysis = ""
    
    if target_shortest_answer is not None:
        target_shortest_soundness_res = TaskManager.check_soundness(
            task_name=args.dataset,
            sample=input_sample,
            response_text=target_shortest_correct_txt
        )
        if target_shortest_soundness_res["decision"] is None:
            target_soundness_check_failed = True

        target_shortest_soundness = not target_shortest_soundness_res["decision"]
        target_shortest_soundness_analysis = target_shortest_soundness_res["analysis"]

    # Log initial responses
    log["initial_target_reponse"] = target_shortest_correct_txt
    log["initial_draft_reponse"] = draft_shortest_correct_txt
    log["target_shortest_soundness"] = target_shortest_soundness
    log["target_shortest_soundness_analysis"] = target_shortest_soundness_analysis
    log["initial_target_utility_rate"] = target_response_utility_rate
    log["initial_draft_utility_rate"] = draft_response_utility_rate

    print("+++++++++++++++++++++++++++++++++++++++++")
    print(f"Target response: {target_shortest_correct_txt}")
    print("+++++++++++++++++++++++++++++++++++++++++")
    print(f"Draft response: {draft_shortest_correct_txt}")
    print("+++++++++++++++++++++++++++++++++++++++++")

    # Validate target response quality
    if target_soundness_check_failed or (target_response_utility_rate <= args.stop_threshold) or (not target_shortest_soundness):
        print("invalid or wrong target response")
        log["invalid"] = True
        log["reason"] = "invalid target response"
        return (target_shortest_correct_txt, [], log)

    # Skip if draft already produces correct answer
    if draft_response_utility_rate >= args.sampling_rate_threshold and args.skip_correct_draft:
        log["steps"] = steps
        log["response"] = draft_shortest_correct_txt
        log["time"] = time.time() - start_time
        log["draft_response_utility_rate"] = draft_response_utility_rate
        log["skip_correct_draft"] = True
        print(draft_shortest_correct_txt)
        return draft_shortest_correct_txt, [], log

    # Initialize generation loop variables
    prev_len = current_sequence.shape[1]
    prev_target_past_key_values = None
    prev_draft_past_key_values = None
    start_index = 0
    step_index = 0
    
    # Main generation loop
    for t in range(max_iter):
        print(f"Step: {t}")
        
        # Check if generation should stop
        if any(stop_token in current_sequence[0][prev_len:] for stop_token in stop_token_ids):
            print(f"Reached stop token, closing...")
            break

        prev_len = current_sequence.shape[1]
        
        # Generate from draft model
        draft_seq, draft_logits, draft_past_key_values = complete_using_generation(
            draft_model,
            tokenizer,
            current_sequence,
            max_new_tokens=spec_len,
            max_seq_len=max_seq_len,
            is_target=False,
            past_key_values=prev_draft_past_key_values,
            start_index=start_index,
            return_dict_in_generate=True,
        )

        n_draft_new_tokens = draft_seq.shape[1] - prev_len

        # Generate from target model
        target_seq, target_logits, target_past_key_values = complete_using_generation(
            target_model,
            tokenizer,
            current_sequence,
            max_new_tokens=spec_len,
            max_seq_len=max_seq_len,
            is_target=True,
            past_key_values=prev_target_past_key_values,
            start_index=start_index,
            return_dict_in_generate=True,
        )

        # Validate if draft tokens match target tokens
        all_varified, n_verified_tokens, directly_pivotal = validate_token(
            draft_seq[-1][prev_len:],
            target_seq[-1][prev_len:],
            draft_logits[0] if draft_logits is not None else None,
            target_logits[0] if target_logits is not None else None,
        )

        # Update current sequence with verified tokens
        current_sequence = draft_seq[:, : prev_len + n_verified_tokens]
        cur_len = current_sequence.shape[1]
        formatted_output += tokenizer.decode(draft_seq[0][prev_len:cur_len])

        if cur_len > max_seq_len:
            print("Exceeded max new tokens")
            log["invalid"] = True
            log["reason"] = "exceeded max new tokens"
            break

        # Manage KV cache
        if args.kv_caching:
            target_past_key_values_accepted = select_kv_cache_portion(
                target_past_key_values, 0, prev_len + n_verified_tokens - 1
            )
            prev_target_past_key_values = target_past_key_values_accepted
            prev_draft_past_key_values = select_kv_cache_portion(
                draft_past_key_values, 0, prev_len + n_verified_tokens - 1
            )
        else:
            target_past_key_values_accepted = None
            target_past_key_values = None
            prev_target_past_key_values = None
            prev_draft_past_key_values = None

        if all_varified:
            print("all varified **************")

        if not all_varified:
            print("not all varified **************")
            print(f"Draft seq: {tokenizer.decode(draft_seq[0][cur_len:])}, Target seq: {tokenizer.decode(target_seq[0][cur_len:])}")
            
            # Initialize record and sample for this step
            record = {}
            sample = {}
            sample["step_index"] = step_index
            record["step_index"] = step_index
            step_index += 1

            passed_seq_target = target_seq

            # Check if utility changed between accepted and rejected sequences
            (
                utility_changed_sampling_result,
                accept_continuation,
                reject_continuation,
                target_past_key_values_accepted,
            ) = utility_changed_sampling(
                target_model,
                draft_model,
                tokenizer,
                args.dataset,
                draft_seq,
                passed_seq_target,
                max_seq_len - draft_seq.shape[1],
                max_seq_len,
                init_len,
                prev_len,
                num_samples=sampling_num,
                self_supervised=self_supervised,
                sampling_rate_threshold=sampling_rate_threshold,
                original_ground_truth=ground_truth,
                num_beams=num_beams,
                target_past_key_values_rejected=target_past_key_values,
                target_past_key_values_accepted=target_past_key_values_accepted,
                sample=input_sample,
                max_extra_len_ratio=max_extra_len_ratio,
                self_correction_check_mode=self_correction_check_mode,
                mlc=mlc,
                record=record,
            )

            # Get shortest correct answers from continuations
            accept_shortest_correct, len_correct_accepts = get_shortest_with_correct_answer(
                accept_continuation,
                ground_truth,
                init_len,
                tokenizer,
                dataset=args.dataset,
                sample=input_sample,
            )
            reject_shortest_correct, len_correct_rejects = get_shortest_with_correct_answer(
                reject_continuation,
                ground_truth,
                init_len,
                tokenizer,
                dataset=args.dataset,
                sample=input_sample,
            )

            record["len_accept_shortest"] = len_correct_accepts
            record["len_reject_shortest"] = len_correct_rejects
            record["len_shortest_accept"] = len(accept_shortest_correct) if accept_shortest_correct is not None else 0
            record["len_shortest_reject"] = len(reject_shortest_correct) if reject_shortest_correct is not None else 0

            txt_accept_shortest_correct = tokenizer.decode(
                accept_shortest_correct[init_len:] if accept_shortest_correct is not None else accept_continuation[0][init_len:],
                skip_special_tokens=True
            )
            txt_reject_shortest_correct = tokenizer.decode(
                reject_shortest_correct[init_len:] if reject_shortest_correct is not None else reject_continuation[0][init_len:],
                skip_special_tokens=True
            )

            # Check soundness of accept continuation
            accept_shortest_soundness = None
            accept_soundness_check_failed = False
            if accept_shortest_correct is not None:
                accept_shortest_soundness_res = TaskManager.check_soundness(
                    task_name=args.dataset,
                    sample=input_sample,
                    response_text=txt_accept_shortest_correct
                )
                if accept_shortest_soundness_res["decision"] is None:
                    record["soundness_check_failed"] = True
                    accept_soundness_check_failed = True
                    
                accept_shortest_soundness = not accept_shortest_soundness_res["decision"]
                record["accept_shortest_soundness_analysis"] = accept_shortest_soundness_res["analysis"]
                print(f"Accept Shortest Soundness: {accept_shortest_soundness}, Analysis: {accept_shortest_soundness_res['analysis']}")

            record["accept_shortest_soundness"] = accept_shortest_soundness

            # Check soundness of reject continuation
            reject_shortest_soundness = None
            reject_soundness_check_failed = False
            if reject_shortest_correct is not None:
                reject_shortest_soundness_res = TaskManager.check_soundness(
                    task_name=args.dataset,
                    sample=input_sample,
                    response_text=txt_reject_shortest_correct
                )
                if reject_shortest_soundness_res["decision"] is None:
                    record["soundness_check_failed"] = True
                    reject_soundness_check_failed = True
                reject_shortest_soundness = not reject_shortest_soundness_res["decision"]
                record["reject_shortest_soundness_analysis"] = reject_shortest_soundness_res["analysis"]
                print(f"Reject Shortest Soundness: {reject_shortest_soundness}, Analysis: {reject_shortest_soundness_res['analysis']}")
            
            record["reject_shortest_soundness"] = reject_shortest_soundness


            target_response_utility_rate = get_utility_rate_by_dataset(
                reject_continuation,
                ground_truth,
                init_len,
                tokenizer,
                dataset=args.dataset,
                sample=input_sample,
                record=record,
                target_model=target_model,
                self_supervised=self_supervised,
                max_extra_len=max_extra_len_ratio,
                prev_len=prev_len,
                self_correction_check_mode=self_correction_check_mode,
                mlc=mlc
            )

            # Determine if token is pivotal
            print(f"---------------------------------")
            print(f"utility_changed_sampling_result: {utility_changed_sampling_result}")
            print(f"accept_shortest_soundness: {accept_shortest_soundness}")

            not_pivotal = not utility_changed_sampling_result and accept_shortest_soundness
            pivotal = not not_pivotal
            print(f"Pivotal: {pivotal}")

            prev_target_past_key_values = None

            # Update current sequence based on trajectory following mode
            if args.follow_target_trajectory:
                # Following target trajectory: always append target token
                if not pivotal:
                    formatted_output += f"{Style.RESET_ALL}{Fore.GREEN}{strike(tokenizer.decode(draft_seq[0][cur_len:]))}{Style.RESET_ALL}{Fore.BLUE}{tokenizer.decode(target_seq[0][cur_len: cur_len+1])}{Style.RESET_ALL}"
                else:
                    formatted_output += f"{Style.RESET_ALL}{Fore.RED}{strike(tokenizer.decode(draft_seq[0][cur_len:]))}{Style.RESET_ALL}{Fore.BLUE}{tokenizer.decode(target_seq[0][cur_len: cur_len+1])}{Style.RESET_ALL}"

                current_sequence = target_seq[:, : current_sequence.shape[1] + 1]
                if args.kv_caching:
                    prev_target_past_key_values = select_kv_cache_portion(target_past_key_values, 0, prev_len + n_verified_tokens)
                    prev_draft_past_key_values = select_kv_cache_portion(draft_past_key_values, 0, prev_len + n_verified_tokens)
            else:
                # Not following target trajectory: use draft if not pivotal, target if pivotal
                if not pivotal:
                    current_sequence = draft_seq
                    formatted_output += f"{Fore.BLUE}{tokenizer.decode(draft_seq[0][cur_len:])}{Style.RESET_ALL}"
                    if args.kv_caching:
                        prev_target_past_key_values = select_kv_cache_portion(target_past_key_values_accepted, 0, prev_len + spec_len - 1)
                        prev_draft_past_key_values = select_kv_cache_portion(draft_past_key_values, 0, prev_len + spec_len - 1)
                else:
                    formatted_output += f"{Style.RESET_ALL}{Fore.BLUE}{strike(tokenizer.decode(draft_seq[0][cur_len:]))}{Style.RESET_ALL}{Fore.RED}{tokenizer.decode(target_seq[0][cur_len: cur_len+1])}{Style.RESET_ALL}"
                    current_sequence = target_seq[:, : current_sequence.shape[1] + 1]
                    if args.kv_caching:
                        prev_target_past_key_values = select_kv_cache_portion(target_past_key_values, 0, prev_len + n_verified_tokens)
                        prev_draft_past_key_values = select_kv_cache_portion(draft_past_key_values, 0, prev_len + n_verified_tokens)

            # Record step information
            record["verified"] = False
            record["pivotal"] = pivotal
            record["draft_seq"] = tokenizer.decode(draft_seq[0][cur_len:])
            record["target_seq"] = tokenizer.decode(target_seq[0][cur_len:])
            record["draft_seq_token"] = draft_seq[0][cur_len:].tolist()
            record["target_seq_token"] = target_seq[0][cur_len:].tolist()
            record["accept_cont"] = txt_accept_shortest_correct
            record["reject_cont"] = txt_reject_shortest_correct

            if pivotal:
                print(f"Accept Continuation: {txt_accept_shortest_correct}")
                print(f"Reject Continuation: {txt_reject_shortest_correct}")

            print(f"Draft seq: {tokenizer.decode(draft_seq[0][cur_len:])}, Target seq: {tokenizer.decode(target_seq[0][cur_len:])}")
            print(f"Pivotal: {pivotal}")
            print(formatted_output)
            print("2-------------------------------------")

            # Filter pivotal tokens: keep high-quality ones that lead to low prob_accepted and unsound reasoning
            # Keep all non-pivotal tokens as they offer high prob_accepted and lead to sound reasoning
            if pivotal:
                keep = (
                    record["prob_accepted"] < record["prob_rejected"]  # Prob of reaching good solution is lower
                    and not accept_shortest_soundness  # Accepting the token cannot reach answer in a sound way
                )
                record["skipped"] = not keep
                sample["skipped"] = not keep
            
            sample["accept_soundness"] = accept_shortest_soundness
            sample["accept_soundness_check_failed"] = accept_soundness_check_failed
            sample["reject_soundness"] = reject_shortest_soundness
            sample["reject_soundness_check_failed"] = reject_soundness_check_failed
            steps.append(record)

            # Stop if both continuations are not good
            if (pivotal or target_response_utility_rate <= args.stop_threshold) and not reject_shortest_soundness:
                log["invalid"] = True
                log["reason"] = "Both accept and reject continuations are not good"
                log["final_reject_cont"] = txt_reject_shortest_correct
                log["final_accept_utility_rate"] = target_response_utility_rate
                print("Invalid or wrong target answer")
                print("Accept Continuation: ", tokenizer.decode(accept_continuation[0][init_len:], skip_special_tokens=True))
                print("Reject Continuation: ", tokenizer.decode(reject_continuation[0][init_len:], skip_special_tokens=True))
                break
                
            # Prepare sample for dataset
            sample["prompt"] = prompt
            sample["generated"] = tokenizer.decode(current_sequence[0][init_len:])
            sample["tokens"] = draft_seq[0].tolist()
            sample["label"] = generate_label(draft_seq[0], n_draft_new_tokens, n_verified_tokens, pivotal)
            sample["data_index"] = data_index
            sample["prob_accepted"] = record["prob_accepted"]
            sample["prob_rejected"] = record["prob_rejected"]
            if pivotal:
                sample["accept_cont_tokens"] = txt_accept_shortest_correct
                sample["reject_cont_tokens"] = txt_reject_shortest_correct

            # Save sample to dataset
            if not rank:
                save_line_jsonlines(output_dataset_dir, sample)

        # Check stop condition
        if any(stop_token in current_sequence[0][prev_len:] for stop_token in stop_token_ids):
            print(f"Closing 1, {tokenizer.decode(current_sequence[0][init_len:])}")
            break
        
        # Save intermediate logs
        if not rank:
            log["steps"] = steps
            print(f"Ranking {rank}, saving logs to {log_dir}")
            with open(log_dir, "w") as f:
                f.write(json.dumps(log, indent=4))


    print(f"Saved response: {tokenizer.decode(current_sequence[0][init_len:])}")
    response = tokenizer.decode(current_sequence[0][init_len:])
    log["steps"] = steps
    log["response"] = response
    log["time"] = time.time() - start_time
    log["formatted_output"] = formatted_output
    return response, log


def skip_sample(train_data, data_index, dataset):
    """Check if a sample should be skipped based on dataset-specific criteria"""
    if dataset == "mbpp" and data_index == 678:
        print("Skipping sample 678")
        return True
    
    if dataset == "apps":
        sample = train_data[data_index]
        print(sample.keys())
        if "input_output" not in sample:
            print(f"Skipping sample {data_index} because it does not have input_output")
            return True
        try:
            json.loads(sample["input_output"])
        except:
            print(f"Skipping sample {data_index} because input_output does not load")
            return True
    elif dataset == "mbpp":
        sample = train_data[data_index]
        if generate_prompt_mbpp(sample) is None:
            print(f"Skipping sample {data_index} because it does not have input_output")
            return True
    
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training dataset using speculative decoding with draft and target models")
    
    # Data processing arguments
    parser.add_argument("--part", type=int, default=1, help="Which part of the dataset to process")
    parser.add_argument("--chunk_size", type=int, default=3000, help="Size of each chunk")
    parser.add_argument("--data_start_idx", type=int, default=None, help="Starting index for data processing")
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "apps", "mbpp", "trivia_qa", "math"])
    
    # Model arguments
    parser.add_argument("--target_model_name", type=str, default="Qwen/Qwen3-8B", help="Target model name")
    parser.add_argument("--draft_model_name", type=str, default="Qwen/Qwen3-0.6B", help="Draft model name")
    parser.add_argument("--quantize", action="store_true", help="Use quantization")
    parser.add_argument("--device", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--no_compile", action="store_true", help="Disable model compilation")
    
    # Generation arguments
    parser.add_argument("--spec_len", type=int, default=1, help="Speculative decoding length")
    parser.add_argument("--max_iter", type=int, default=1000, help="Maximum iterations")
    parser.add_argument("--max_seq_len", type=int, default=800, help="Maximum sequence length")
    parser.add_argument("--max_extra_len", type=int, default=500, help="Maximum extra length")
    parser.add_argument("--max_extra_len_ratio", type=float, default=None, help="Maximum extra length ratio")
    parser.add_argument("--no_kv_caching", action="store_true", help="Disable KV caching")
    
    # Sampling arguments
    parser.add_argument("--self_supervised", action="store_true", help="Use self-supervised mode")
    parser.add_argument("--sampling_num", type=int, default=20, help="Number of samples")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search")
    parser.add_argument("--sampling_rate_threshold", type=float, default=None, help="Sampling rate threshold for token rejection")
    parser.add_argument("--stop_threshold", type=float, default=0.3, help="Stop threshold for utility rate")
    parser.add_argument("--skip_threshold", type=float, default=0.3, help="Skip threshold for pivotal tokens")
    parser.add_argument("--ratio_compare_type", type=str, default='mul', choices=['mul', 'add', 'sa'], 
                        help="Comparison type: mul=multiply, add=addition, sa=standalone")
    
    # Quality control arguments
    parser.add_argument("--self_correction_check_mode", type=str, default="rule", help="Self-correction check mode: rule, llm, or None")
    parser.add_argument('--mlc', type=int, default=5, help='Minimum comparison length')
    parser.add_argument("--skip_correct_draft", action="store_true", help="Skip samples where draft is already correct")
    parser.add_argument("--include_wrong_answers", action="store_true", help="Include wrong answers in dataset")
    parser.add_argument('--follow_target_trajectory', action='store_true', help='Always follow target trajectory')
    
    # Output arguments
    parser.add_argument("--save_name", type=str, default="v31_gpt_fast", help="Name for saved outputs")
    parser.add_argument("--gpt_fast", action="store_true", help="Use GPT-fast mode")

    args = parser.parse_args()
    args.kv_caching = not args.no_kv_caching


    # Initialize tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.target_model_name)
    pad_token_id = tokenizer.pad_token_id

    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    # Setup device and model names
    device = f"cuda:{args.device}"
    target_model_name = args.target_model_name
    draft_model_name = args.draft_model_name

    # Configure stop tokens based on model type
    stop_token_ids = [tokenizer.eos_token_id]
    if 'gemma' in target_model_name.lower():
        stop_token_ids.append(106)
    if "qwen" in target_model_name.lower():
        stop_token_ids.append(151645)  # <|im_end|>

    # Configure tensor parallelism
    tensor_parallel_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Load target model
    target_model = LLM(
        model=target_model_name,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        enable_prefix_caching=True,
        max_model_len=args.max_seq_len,
        gpu_memory_utilization=0.70
    )
    print(f"Loaded Target Model: {target_model_name}")

    # Load draft model
    draft_model = LLM(
        model=draft_model_name,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        enable_prefix_caching=True,
        max_model_len=args.max_seq_len,
        gpu_memory_utilization=0.15
    )
    print(f"Loaded Draft Model: {draft_model_name}")
    # Setup data processing range
    start_time = time.time()
    part = args.part
    chunk_size = args.chunk_size
    start_idx = (part - 1) * chunk_size if args.data_start_idx is None else args.data_start_idx
    end_idx = part * chunk_size
    
    # Setup output directories
    output_base_dir = f"./output/{args.save_name}_{args.dataset}_{args.target_model_name.split('/')[-1]}_{args.draft_model_name.split('/')[-1]}/{args.spec_len}/"
    output_dataset_dir = f"{output_base_dir}/dataset_part{part}.jsonl"
    os.makedirs(os.path.dirname(output_dataset_dir), exist_ok=True)

    # Save arguments to JSON file
    args_dict = vars(args)
    with open(f"{output_base_dir}/args.json", "w") as f:
        json.dump(args_dict, f, indent=4)

    # Load dataset
    train_data, test_data = get_dataset(args.dataset)

    # Process samples
    for i in range(start_idx, end_idx):
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(f"Processing sample {i} in part {part}")
        print(f"Rank: {rank}, Device: {device}")
        log_memory_usage(f"Start of sample {i}")
        print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        
        # Clean up memory and set seed
        torch.cuda.empty_cache()
        torch.manual_seed(i)
        data_index = i
        log_dir = f"{output_base_dir}/{data_index}.json"

        # Skip sample if needed
        if skip_sample(train_data, data_index, args.dataset):
            print(f"Skipping sample {data_index} in dataset {args.dataset}")
            continue

        os.makedirs(os.path.dirname(log_dir), exist_ok=True)

        # Generate dataset for this sample
        logs_all = generate_dataset(
            target_model,
            draft_model,
            tokenizer,
            data_index,
            train_data,
            max_iter=args.max_iter,
            spec_len=args.spec_len,
            max_seq_len=args.max_seq_len,
            max_extra_len=args.max_extra_len,
            max_extra_len_ratio=args.max_extra_len_ratio,
            self_supervised=args.self_supervised,
            sampling_num=args.sampling_num,
            sampling_rate_threshold=args.sampling_rate_threshold,
            num_beams=args.num_beams,
            output_dataset_dir=output_dataset_dir,
            self_correction_check_mode=args.self_correction_check_mode,
            mlc=args.mlc,
            log_dir=log_dir
        )
        
        # Save logs
        if not rank:
            print(f"Ranking {rank}, saving logs to {log_dir}")
            with open(log_dir, "w") as f:
                f.write(json.dumps(logs_all, indent=4))
        
        # Memory cleanup after each sample
        del logs_all
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU memory after sample {i}: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
