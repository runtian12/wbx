import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
import re
import transformers
import torch
import time
import argparse
from src.util.gsm8k_helper import *
from colorama import Fore, Style
import tqdm
import wandb
from src.classifiers import TorchMLP, AnchorClassifierExtendedTarget, AnchorClassifierExtendedTargetV2
from src.util.tasks import TaskManager
from src.util.general import get_text
from src.gpt_fast.gpt_fast_utils import setup_gpt_fast, complete_using_generation, get_logits_and_emneddings, logits_to_probs, multinomial_sample_one_no_sync
import random
torch.manual_seed(42)
random.seed(42)

from torch.nn.functional import softmax
import matplotlib.pyplot as plt

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in ("yes","true","t","y","1"):
        return True
    if v in ("no","false","f","n","0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected (true/false).")

def save_mlp_model(model, path):
    torch.save(model.state_dict(), path)


def load_mlp_model(model, path):
    if path is not None:
        model.load_state_dict(torch.load(path))
    return model


def strike(text):
    return "".join(["{}\u0336".format(c) for c in text])


def get_dataset(dataset):
    """Load dataset using TaskManager following dataset_generation pattern"""
    base_paths = {
        "gsm8k": "data/gsm8k",
        "math": "data/math_splits",
    }
    base_path = base_paths.get(dataset, None)
    return TaskManager.load_dataset(dataset, base_path=base_path)


def check_correctness_by_dataset(
    sequence_or_text, polished_reference_txt, init_len=0, tokenizer=None, dataset=None, sample=None, **kwargs
):
    """Check correctness using TaskManager following dataset_generation pattern"""
    return TaskManager.evaluate(
            dataset,
            sequence_or_text,
            polished_reference_txt,
            sample=sample,
            start_idx=init_len,
            tokenizer=tokenizer,
            polish_reference=False,
    )


import time


def polish_answer(answer):
    """
    Enhanced version that extracts the first number from the answer string.
    Uses extract_answer to get text after #### first, then finds the first number.
    Falls back to '' if no number is found.
    """
    
    # Strip whitespace
    answer = answer.strip()
    
    # Pattern to match numbers (including decimals, negatives, and scientific notation)
    number_pattern = r'-?\d*\.?\d+(?:[eE][+-]?\d+)?'
    
    # Find all numbers in the string
    numbers = re.findall(number_pattern, answer)
    
    if numbers:
        try:
            # Return the first number found, converted to float
            return float(numbers[0])
        except ValueError:
            # If conversion fails, fall back to original behavior
            pass
    
    # If no valid number found, return ''
    return ''


def extract_answer(answer):
    return answer.split("####")[-1].strip()


def is_valid_answer(answer):
    return len(answer.split(" ")) == 1



def get_num_valid_tokens(mask):
    num_valid_tokens = 0
    for i in range(len(mask)):
        if mask[i] == 0:
            break
        num_valid_tokens += 1
    return num_valid_tokens



def sp_validate_token(candidate_new_tokens, do_sample=False, target_dist=None, draft_dist=None):
    # Get probabilities from logits
    q = draft_dist
    p = target_dist

    # Get probabilities of selected tokens
    q_i = q[torch.arange(len(candidate_new_tokens)), candidate_new_tokens]
    p_i = p[torch.arange(len(candidate_new_tokens)), candidate_new_tokens]

    # Calculate probability ratio
    probability_ratio = p_i / q_i

    if not do_sample:
        sp_mask = target_dist.argmax(-1) == draft_dist.argmax(-1)
    else:
        r_i = torch.rand_like(probability_ratio)
        sp_mask = r_i <= probability_ratio

    return sp_mask.int(), [p_i, q_i, probability_ratio]


def check_pivotal(
    embeddings,
    classifier,
    classifier_type="linear",
    threshold=0.1,
    candidate_new_tokens=None,
    draft_dist=None,
    target_dist=None,
):
    if classifier_type == 'extended' or classifier_type == 'extended_v2':
        # Compute entropies and reshape to (b, seq_len, 1)
        target_entropy = - torch.sum(target_dist * torch.log(target_dist + 1e-10), dim=-1).unsqueeze(-1)
        draft_entropy = - torch.sum(draft_dist * torch.log(draft_dist + 1e-10), dim=-1).unsqueeze(-1)



        p = target_dist # Shape (batch_size, seq_len, vocab_size)
        q = draft_dist

        p_i = torch.gather(p, 2, candidate_new_tokens.unsqueeze(-1).unsqueeze(0))
        q_i = torch.gather(q, 2, candidate_new_tokens.unsqueeze(-1).unsqueeze(0))


        with torch.no_grad():
            logits = classifier(
                target_hidden=embeddings.float(),
                target_entropy=target_entropy.float(),
                draft_entropy=draft_entropy.float(),
                target_logit=p_i.float(),
                draft_logit=q_i.float(),
            )
    if classifier_type == 'linear' or classifier_type == 'mlp':
        with torch.no_grad():
            logits = classifier(embeddings.float())
            # pivotal_mask = logits.argmax(dim=-1)
    prob_pivotal = softmax(logits, dim=-1)[:, :, 1]
    is_pivotal = (prob_pivotal > threshold).int()
    pivotal_mask = 1 - is_pivotal # if is pivotal then 0, else 1


    return pivotal_mask, prob_pivotal




def generate_answers(
    target_model,
    draft_model,
    tokenizer,
    current_sequence,
    spec_len=2,
    classifier=None,
    n_shot=1,
    do_sample=False,
    threshold=0,
    first_iter=True,
    prob_ratio_threshold=0,
    layer_index=-1,
    max_seq_len=1
):
    n_accepted_tokens_list = []
    n_sp_accepted_tokens_list = []
    n_new_tokens_list = []

    max_new_tokens = 1500

    assert (
        current_sequence.shape[0] == 1
    ), "Support for multiple sequences not implemented yet"

    init_len = current_sequence.shape[1]
    prev_len = current_sequence.shape[1]
    draft_past_key_values = 0
    target_past_key_values = 0
    
    debug_info = []

    while prev_len < max_seq_len:


        draft_response, draft_new_probs, _ = complete_using_generation(
            draft_model,
            tokenizer,
            current_sequence,
            max_new_tokens=spec_len,
            max_seq_len=max_seq_len,
            is_target=False,
            past_key_values=draft_past_key_values,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        new_tokens = draft_response[0][prev_len:]
        n_new_tokens = new_tokens.shape[0]

        assert (
            new_tokens.shape[0] == draft_new_probs.shape[1]
        ), f"New tokens shape: {new_tokens.shape}, logits shape: {draft_new_probs.shape}"

        # print(f"init len: {init_len}, prev len: {prev_len}, draft_response len: {draft_response.shape[1]} ,n new tokens: {n_new_tokens}, past_key_values: {target_past_key_values})")
        target_logits, target_embeddings, target_past_key_values = get_logits_and_emneddings(
            target_model, draft_response, is_target=True, past_key_values=target_past_key_values, max_seq_len=max_seq_len, layer_index=layer_index
        )
        target_validation_logits = target_logits[:, -n_new_tokens - 1 : -1, :]

        target_logits = target_logits[:, -n_new_tokens -1 :, :]

        new_token_embeddings = target_embeddings[:, -n_new_tokens :, :]

        assert (
            target_logits.shape[1] == prev_len + n_new_tokens or target_logits.shape[1] == n_new_tokens + 1
        ), f"Target logits shape: {target_logits.shape}, prev_len: {prev_len}, n_new_tokens: {n_new_tokens}"

        target_dist_sampling = logits_to_probs(target_logits, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
        if do_sample:
            target_dist_orig_validation_tokens = target_validation_logits.softmax(dim=-1) # Needed for Extended classifier
        else:
            target_dist_orig_validation_tokens = target_dist_sampling[:, :-1, :]
        target_dist_sampling_validation_tokens = target_dist_sampling[:, :-1, :]
        draft_dist_sampling_validation_tokens  = draft_new_probs
        # draft_dist = draft_new_logits.softmax(dim=-1)

        sp_mask, stats = sp_validate_token(
            new_tokens, draft_dist=draft_dist_sampling_validation_tokens[0], target_dist=target_dist_sampling_validation_tokens[0], do_sample=do_sample
        )
        pivotal_mask, prob = check_pivotal(
            new_token_embeddings,
            classifier,
            threshold=threshold,
            candidate_new_tokens=new_tokens,
            draft_dist=draft_dist_sampling_validation_tokens, # We don't have the default lgoits here, Extended classifier V2 does not use this
            target_dist=target_dist_orig_validation_tokens,
            classifier_type=args.classifier_type,
        )
        prob = prob[0]

        # Combine SP mask and pivotal mask
        pivotal_mask = pivotal_mask[0]
        final_mask = sp_mask | pivotal_mask
        p_i_orig =  target_dist_orig_validation_tokens[0][torch.arange(len(new_tokens)), new_tokens]
        if do_sample:
            extra_mask = p_i_orig >= prob_ratio_threshold
        else:
            extra_mask = stats[2] >= prob_ratio_threshold
        final_mask = final_mask & extra_mask
        n_sp_valid_tokens = get_num_valid_tokens(sp_mask)
        n_valid_tokens = get_num_valid_tokens(final_mask)


        current_sequence = draft_response[:, : prev_len + n_valid_tokens]


        # Get next token from target model
        if not tokenizer.eos_token_id in current_sequence[0][init_len:]:
            # Get next token from target model
            if do_sample:
                if n_valid_tokens < n_new_tokens:
                    p_n_plus_1 = target_dist_sampling_validation_tokens[0][n_valid_tokens]
                    q_n_plus_1 = draft_dist_sampling_validation_tokens[0][n_valid_tokens]
                    p_prime = torch.clamp((p_n_plus_1 - q_n_plus_1), min=0)
                    p_prime.div_(p_prime.sum())
                else:
                    p_prime = target_dist_sampling[0][n_valid_tokens]
                next_token = multinomial_sample_one_no_sync(p_prime)[0]
            else:
                # next_token = target_logits[0][n_valid_tokens].argmax(dim=-1)
                next_token = target_dist_sampling[0][n_valid_tokens].argmax(dim=-1)

            current_sequence = torch.cat(
                [current_sequence, next_token.unsqueeze(0).unsqueeze(0)], dim=1
            )

        if args.debug:
            # visualize the pivotal mask on tokens
            # Decode new tokens and colorize based on pivotal mask
            p_i, q_i, ratio = stats[0], stats[1], stats[2]
            token_texts = tokenizer.convert_ids_to_tokens(new_tokens)

            # --- Compact Visualization ---
            col_width = 15

            # Prepare data rows
            row_tokens = [f"{token:<{col_width}}" for token in token_texts]
            row_p = [f"{p.item():<{col_width}.4f}" for p in p_i]
            row_q = [f"{q.item():<{col_width}.4f}" for q in q_i]

            row_ratio_str = []
            for r in ratio:
                color = Fore.GREEN if r.item() >= 1.0 else Fore.RED
                s = f"{r.item():.4f}"
                # Manually pad to account for color codes
                padding = " " * (col_width - len(s))
                row_ratio_str.append(f"{color}{s}{Style.RESET_ALL}{padding}")

            row_prob_str = []
            for p in prob:
                color = Fore.GREEN if p.item() <= threshold else Fore.RED
                s = f"{p.item():.4f}"
                # Manually pad to account for color codes
                padding = " " * (col_width - len(s))
                row_prob_str.append(f"{color}{s}{Style.RESET_ALL}{padding}")

            row_target_tokens = []
            target_token_texts = tokenizer.convert_ids_to_tokens(torch.argmax(target_dist_orig_validation_tokens, dim=-1)[0])
            for target_token_text in target_token_texts:
                color = Fore.RED
                # token_text = tokenizer.decode(torch.argmax(target_dist_sampling_validation_tokens[0][i]))
                s = f"{target_token_text:<{col_width}}"
                padding = " " * (col_width - len(s))
                row_target_tokens.append(f"{color}{s}{Style.RESET_ALL}{padding}")

            # Determine replacement tokens
            row_replace = [f"{'':<{col_width}}"] * n_new_tokens
            if n_valid_tokens < n_new_tokens:
                next_token_text = tokenizer.decode(next_token)
                row_replace[n_valid_tokens] = f"{next_token_text:<{col_width}}"
                if n_new_tokens > n_valid_tokens + 1:
                    for i in range(n_valid_tokens + 1, n_new_tokens):
                        row_replace[i] = f"{'(discarded)':<{col_width}}"
            
            row_p_i_orig = [f"{p.item():<{col_width}.4f}" for p in p_i_orig]

            debug_info.append({
                "tokens": row_tokens,
                "p (SP)": row_p,
                "q (SP)": row_q,
                "ratio (SP)": row_ratio_str,
                "prob (Pivotal)": row_prob_str,
                "replacement": row_replace,
                "target tokens": row_target_tokens,
                "p_i_orig": row_p_i_orig,
            })




        prev_len = current_sequence.shape[1]


        n_sp_accepted_tokens_list.append(n_sp_valid_tokens)
        n_accepted_tokens_list.append(n_valid_tokens)
        n_new_tokens_list.append(n_new_tokens)

        if tokenizer.eos_token_id in current_sequence[0][init_len:]:
            break

        if current_sequence.shape[1] >= max_seq_len:
            print("Max length reached")
            break

        target_past_key_values = current_sequence.shape[1] - 2
        draft_past_key_values = current_sequence.shape[1] - 2

    uasp_reponse = current_sequence[0][init_len:max_seq_len]

    return (
        uasp_reponse,
        n_accepted_tokens_list,
        n_sp_accepted_tokens_list,
        n_new_tokens_list,
        debug_info,
    )

parser = argparse.ArgumentParser()
parser.add_argument("--num_samples", type=int, default=None)
parser.add_argument("--classifier_ckp", type=str, default=None)
parser.add_argument("--version", type=str, default="v20_7_extended")
parser.add_argument("--target_model_name", type=str, default="Qwen/Qwen3-8B")
parser.add_argument("--draft_model_name", type=str, default="Qwen/Qwen3-0.6B")
parser.add_argument('--max_iter', type=int, default=32000)
parser.add_argument('--data_split', type=str, default='test')
parser.add_argument('--dataset', type=str, default='gsm8k', choices=['gsm8k', 'math', 'mbpp', 'apps', 'trivia_qa', 'aime24', 'aime25'], help='Dataset to use for evaluation')
parser.add_argument('--spec_len', type=int, default=10)
parser.add_argument('--threshold', type=float, default=0)
parser.add_argument('--prob_threshold', type=float, default=0)
parser.add_argument('--classifier_type', type=str, choices=['mlp', 'linear', 'extended', 'extended_v2'], default='extended', help='Type of classifier to use')
parser.add_argument('--layer_index', type=int, default=-8, help='Layer index to use for embeddings. -1 for last layer.')
parser.add_argument('--debug', action='store_true', help='Enable debug mode for more verbose output')
parser.add_argument("--max_seq_len", type=int, default=32000)
# parser.add_argument('--enable_thinking', action='store_true', help='Enable thinking mode in tokenizer')
parser.add_argument(
    '--enable_thinking',
    type=str2bool,
    default=False,  # Explicitly set the default, which 'store_true' did implicitly
    help='Enable thinking mode in tokenizer'
)
parser.add_argument('--runner', type=str, choices=['gpt-fast', 'hf'], default='gpt-fast', help='Runner to use for decoding')
parser.add_argument('--quantize', type=str2bool, default=False, help='Use quantization for the target model')
parser.add_argument('--compile', type=str2bool, default=False, help='Use TorchScript to compile the model')
parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for sampling (0.0 = greedy)')
parser.add_argument('--top_k', type=int, default=None, help='Top-k sampling parameter')
parser.add_argument('--top_p', type=float, default=None, help='Top-p (nucleus) sampling parameter')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--use_draft', type=str2bool, default=False, help='Use draft model for generation')

args = parser.parse_args()
print(f"Enabled thinking mode: {args.enable_thinking}")


num_samples = args.num_samples
max_iter = args.max_iter
spec_len = args.spec_len
max_seq_len = args.max_seq_len
do_sample = args.temperature > 0.0

target_model_name = args.target_model_name
draft_model_name = args.draft_model_name


tokenizer = transformers.AutoTokenizer.from_pretrained(
    target_model_name, 
)



if args.runner == 'hf':
    if "70b" in target_model_name.lower():
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_8bit=True,
            # llm_int8_enable_fp32_cpu_offload=True,
        )
        print(f"Using 8-bit quantization for {target_model_name}")
    else:
        bnb_config = None
        print(f"Not using quantization for {target_model_name}")
    target_model = transformers.AutoModelForCausalLM.from_pretrained(
            target_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=bnb_config,
        )

    draft_model = transformers.AutoModelForCausalLM.from_pretrained(
        draft_model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        # quantization_config=bnb_config if args.quantize else None,
    )
    device = f'cuda'
    rank = 0
    input_dim = target_model.config.hidden_size
    device = target_model.device

elif args.runner == 'gpt-fast':
    target_model, draft_model, rank, device = setup_gpt_fast(
        args.target_model_name, args.draft_model_name, tokenizer, compile=args.compile, max_seq_len=max_seq_len)
    input_dim = target_model.config.dim * torch.cuda.device_count() # TODO: clean this up
    if rank is None:
        rank = 0
    # input_dim = target_model.config.dim

# Load dataset using TaskManager
dataset_name = args.dataset
train_data, test_data = get_dataset(dataset_name)
if args.data_split == "train":
    data = train_data
elif args.data_split == "test":
    data = test_data
else:
    raise ValueError("Invalid data split")

if num_samples is None:
    num_samples = len(data)

# thresholds = [0.1]
output_dim = 2
# input_dim = target_model.config.dim * torch.cuda.device_count() # TODO: clean this up
print(f"Rank: {rank}, Device: {device}, Input dim: {input_dim}, Output dim: {output_dim}")
if args.classifier_type == 'linear':
    hidden_dim = []
    classifier = TorchMLP(input_dim, hidden_dim, output_dim).to(target_model.device)
elif args.classifier_type == 'mlp':
    hidden_dim = [400, 200]
    classifier = TorchMLP(input_dim, hidden_dim, output_dim).to(target_model.device)
elif args.classifier_type == 'extended':
    classifier = AnchorClassifierExtendedTarget(
        target_hidden_dim=input_dim,
        hidden_dim=128,
        t_embed=64, 
        s_embed=64,
    ).to(device)
elif args.classifier_type == 'extended_v2':
    classifier = AnchorClassifierExtendedTargetV2(
        target_hidden_dim=input_dim,
        hidden_dim=128,
        t_embed=64, 
        s_embed=16,
    ).to(device)
else:
    raise ValueError("Invalid classifier type. Choose 'mlp' or 'linear'.")
classifier.eval()
classifier_path = args.classifier_ckp
print(f"Loading v100 model: {classifier_path}")
classifier = load_mlp_model(classifier, classifier_path)


if rank == 0:
    wandb.init(project="utility-aware-sd-reprod", name=f"{args.version}_{dataset_name}_{args.target_model_name.split('/')[-1]}_target_{args.threshold}_{args.prob_threshold}", config=args)

n_total = 0
n_correct_uasp = 0
n_response_tokens = 0
all_uasp_accepted_tokens = []
all_sp_accepted_tokens = []
all_n_new_tokens = []
threshold = args.threshold
prob_threshold = args.prob_threshold
# n_same = 0
total_infer_time = 0

times = []

if threshold == 0 and prob_threshold != 0:
    # Exit
    print("Threshold is 0 and prob_threshold is not 0, exiting.")
    exit(0)


# for i in tqdm.tqdm(range(1312,1313)):
for i in tqdm.tqdm(range(num_samples)):
    torch.manual_seed(args.seed + i)
# for i in tqdm.tqdm(range(5)):
    print("---------------------")
    print(f"Spec len: {spec_len}")
    print(f"Threshold: {threshold}")
    print(f"Prob threshold: {prob_threshold}")
    print(f"Layer index: {args.layer_index}")
    start_time = time.time()
    data_index = i
    print(f"Data index: {data_index}")
    
    # Prepare sample using TaskManager
    current_sequence, ground_truth, prompt, sample = TaskManager.prepare_sample(
        dataset_name,
        data,
        data_index,
        tokenizer,
        enable_thinking=args.enable_thinking
    )


        # data, data_index, tokenizer, dataset_name, enable_thinking=args.enable_thinking
    # )
    start_infer_time = time.time()

    if args.use_draft:
        draft_response, draft_new_probs, _ = complete_using_generation(
            draft_model,
            tokenizer,
            current_sequence,
            max_new_tokens=max_seq_len,
            max_seq_len=max_seq_len,
            is_target=True,
            past_key_values=0,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            return_probs=False,
        )
        init_len = current_sequence.shape[1]
        uasp_reponse = draft_response[0][init_len:max_seq_len]
        n_accepted_tokens_list = [uasp_reponse.shape[0]]
        n_sp_accepted_tokens_list = [uasp_reponse.shape[0]]
        n_new_tokens_list = [uasp_reponse.shape[0]]
        debug_info = []
    else:
        (
            uasp_reponse,
            n_accepted_tokens_list,
            n_sp_accepted_tokens_list,
            n_new_tokens_list,
            debug_info
        ) = generate_answers(
            target_model,
            draft_model,
            tokenizer,
            current_sequence.to(device),
            spec_len=spec_len,
            classifier=classifier,
            threshold=threshold,
            prob_ratio_threshold=prob_threshold,
            do_sample=do_sample,
            layer_index=args.layer_index,
            max_seq_len=max_seq_len
        )


    infer_time = time.time() - start_infer_time
    total_infer_time += infer_time

    n_total += 1
    n_response_tokens += len(uasp_reponse)

    if 'qwen' in args.target_model_name.lower() and args.enable_thinking:
        # parsing thinking content
        # find idx of 151668 token inside uasp_reponse
        thinking_token_id = 151668
        if thinking_token_id in uasp_reponse:
            idx = (uasp_reponse == thinking_token_id).nonzero(as_tuple=True)[0][-1].item()
            print(idx)
            print(f"Found thinking token at index {idx}, trimming response.")
            uasp_reponse = uasp_reponse[idx+1:]
        else:
            if rank == 0:
                wandb.log({
                    'sample_idx': i,
                    'sample_accuracy': 0,
                    'inference_time': infer_time,
                    'sample_skipped': True,
                    "n_response_tokens": n_response_tokens,
                })
            continue
    
    # Get ground truth answer using TaskManager
    ground_truth_ans = TaskManager.polish_ground_truth(dataset_name, ground_truth)


    
    # Get generated answer using TaskManager
    uasp_response_text = get_text(uasp_reponse, tokenizer, skip_special_tokens=True)

    # Replace \boxedboxed with \boxed
    uasp_response_text = uasp_response_text.replace("\\boxedboxed", "\\boxed")
    
    # Get question text using TaskManager
    question_text = TaskManager.get_question(dataset_name, sample)
    print(f"Question: {question_text}")
    print(f"Ground truth: {ground_truth_ans}")


    # Use dataset-agnostic evaluation
    is_correct = TaskManager.evaluate(
        dataset_name,
        uasp_response_text,
        ground_truth,
        sample=sample,
        start_idx=0,
        tokenizer=tokenizer,
        polish_reference=True
    )
    if not is_correct:
        if args.debug:
            for df in debug_info:
                row_tokens = df["tokens"]
                row_p = df["p (SP)"]
                row_q = df["q (SP)"]
                row_ratio_str = df["ratio (SP)"]
                row_prob_str = df["prob (Pivotal)"]
                row_replace = df["replacement"]
                row_target_tokens = df["target tokens"]
                row_p_i_orig = df["p_i_orig"]
                # Print table
                print("\n" + "="*40)
                print("Token-wise analysis (Iteration)")
                print("="*40)

                print(f"{'Tokens':<15}: {''.join(row_tokens)}")
                print(f"{'p (SP)':<15}: {''.join(row_p)}")
                print(f"{'q (SP)':<15}: {''.join(row_q)}")
                print(f"{'Ratio (SP)':<15}: {''.join(row_ratio_str)}")
                print(f"{'Prob (Pivotal)':<15}: {''.join(row_prob_str)}")
                print(f"{'Replacement':<15}: {''.join(row_replace)}")
                print(f"{'Target Tokens':<15}: {''.join(row_target_tokens)}")
                print(f"{'p_i_orig':<15}: {''.join(row_p_i_orig)}")
                print("-" * 40 + "\n")
        print(f"Generated response: {uasp_response_text}")

    if is_correct:
        n_correct_uasp += 1

    all_sp_accepted_tokens.append(n_sp_accepted_tokens_list)
    all_uasp_accepted_tokens.append(n_accepted_tokens_list)
    all_n_new_tokens.append(n_new_tokens_list)

    sp_acceptance_rate = sum([sum(x) for x in all_sp_accepted_tokens]) / sum([sum(x) for x in all_n_new_tokens])
    uasp_acceptance_rate = sum([sum(x) for x in all_uasp_accepted_tokens]) / sum([sum(x) for x in all_n_new_tokens])


    print(f"SP acceptance rate: {sp_acceptance_rate}")
    print(f"UASP acceptance rate: {uasp_acceptance_rate}")
    print(f'accuracy: {n_correct_uasp/n_total}')

    iter_time = time.time() - start_time
    print(f"Iteration time: {iter_time}")
    times.append(iter_time)
    

    sp_acceptance_rate = sum([sum(x) for x in all_sp_accepted_tokens]) / sum([sum(x) for x in all_n_new_tokens])
    uasp_acceptance_rate = sum([sum(x) for x in all_uasp_accepted_tokens]) / sum([sum(x) for x in all_n_new_tokens])
    print(f"n correct uasp: {n_correct_uasp}, n total: {n_total}")
    print(f"sample_new_tokens: {sum(n_new_tokens_list)}, sample_accepted_tokens: {sum(n_accepted_tokens_list)}, sample_acceptance_rate: {sum(n_accepted_tokens_list)/sum(n_new_tokens_list)}")
    if rank == 0:
        wandb.log({
            'sample_idx': i,
            'sample_accuracy': is_correct,
            'sample_accepted_tokens': sum(n_accepted_tokens_list),
            'sample_new_tokens': sum(n_new_tokens_list),
            'sample_acceptance_rate': sum(n_accepted_tokens_list)/sum(n_new_tokens_list),
            "current_accuracy": n_correct_uasp/n_total,
            "current_sp_acceptance_rate": sp_acceptance_rate,
            "current_uasp_acceptance_rate": uasp_acceptance_rate,
            "n_response_tokens": n_response_tokens,
            'inference_time': infer_time,
            'current_total_inference_time': total_infer_time,
        })

if rank == 0:
    wandb.log({
            "sp_acceptance_rate": sp_acceptance_rate,
            "uasp_acceptance_rate": uasp_acceptance_rate,
            "accuracy": n_correct_uasp/n_total,
            'threshold': threshold,
            'prob_threshold': prob_threshold,
            'total_inference_time': total_infer_time,
        })
