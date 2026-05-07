# Train classifier to identify pivotal tokens
# Includes analysis of effect of prob_accepted and prob_rejected thresholds

import os
import json
import time
import pickle
import copy
from typing import Optional, Union, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
import transformers
from datasets import Dataset
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification

from src.classifiers import TorchMLP, AnchorClassifierExtendedTarget, AnchorClassifierExtendedTargetV2
from src.util.json_io import load_jsonlines
from src.util.gsm8k_helper import nshot_chats


class Args:
    """Configuration for training classifier"""
    target_model_name = "Qwen/Qwen3-8B"
    draft_model_name = "Qwen/Qwen3-0.6B"
    source_model_name = None  # Optional: source model used for dataset generation
    dataset = "gsm8k"  # Dataset type for proper prompt formatting
    batch_size = 100000
    n_epochs = 2000
    lr = 0.00001
    seed = 42
    data_base_dir = "./output/vllm_v20_10_gsm8k_Qwen3-8B_Qwen3-0.6B/1"
    identifier = None
    data_base_dir_2 = None
    identifier_2 = None
    load_act = False
    save_head = True
    layer_index = -8
    remove_duplicates = True
    remove_duplicates_db2 = True
    max_seq_len = 6
    mlp_mode = 'last_token'  # Options: last_token, last_and_first_token, all_tokens
    classifier_type = "extended_v2"  # Options: mlp, linear, extended, extended_v2
    collect_layer_indices = [-1, -8, -10]
    quantize = True


def eval(model, test_loader, thresholds, loss_fn=None):
    """Evaluate model on test set with multiple classification thresholds"""
    model.eval()
    device = next(model.parameters()).device
    metrics_output_dict = {}
    all_outputs = []
    all_labels = []
    all_data_index = []
    all_step_index = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            input_batch = {}
            labels = None
            data_index = None
            step_index = None
            
            # Move batch to device
            for k, v in batch.items():
                batch[k] = v.to(device)

            # Separate labels and metadata from input
            for k, v in batch.items():
                if k == "labels":
                    labels = v
                elif k == "data_index":
                    data_index = v
                elif k == "step_index":
                    step_index = v
                else:
                    input_batch[k] = v

            outputs = model(**input_batch)

            # Calculate loss if loss function is provided
            if loss_fn is not None:
                loss = loss_fn(outputs, labels)
                total_loss += loss.item()
                num_batches += 1

            output_proba = torch.softmax(outputs, dim=1)[:, 1]
            all_outputs.extend(output_proba.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            if data_index is not None:
                all_data_index.extend(data_index.cpu().numpy())
            if step_index is not None:
                all_step_index.extend(step_index.cpu().numpy())

    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)
    all_data_index = np.array(all_data_index) if all_data_index else None
    all_step_index = np.array(all_step_index) if all_step_index else None

    # Calculate metrics for each threshold
    for threshold in thresholds:
        predictions = all_outputs > threshold
        threshold_metrics = {}
        threshold_metrics["f1"] = f1_score(all_labels, predictions)
        threshold_metrics["precision"] = precision_score(all_labels, predictions)
        threshold_metrics["recall"] = recall_score(all_labels, predictions)
        threshold_metrics["accuracy"] = accuracy_score(all_labels, predictions)

        # Add wrongly classified data_index and step_index
        wrong_predictions = predictions != all_labels
        if all_data_index is not None and all_step_index is not None:
            wrong_data_index = all_data_index[wrong_predictions]
            wrong_step_index = all_step_index[wrong_predictions]
            threshold_metrics["wrong_data_index"] = wrong_data_index.tolist()
            threshold_metrics["wrong_step_index"] = wrong_step_index.tolist()

        # Add test loss to metrics
        if loss_fn is not None and num_batches > 0:
            threshold_metrics["test_loss"] = total_loss / num_batches

        metrics_output_dict[threshold] = threshold_metrics

    return metrics_output_dict


def get_logits_and_embeddings(model, input_ids, attention_mask=None, past_key_values=None):
    """Get model logits and hidden states"""
    with torch.no_grad():
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            return_dict=True,
            output_hidden_states=True,
        )
        logits = outputs.logits
        hidden_states = outputs.hidden_states
    return logits, hidden_states

def convert_act_dict_to_tensors(act_dict, layer_index, device="cpu", mlp_mode='all_tokens'):
    """
    Convert activation dictionary from numpy arrays to PyTorch tensors.
    Process hidden states according to mlp_mode: last_token, last_and_first_token, or all_tokens.
    """
    # Convert to torch tensors from numpy arrays
    draft_hidden_tensor = torch.from_numpy(act_dict["draft_hidden_state"][layer_index]).float().to(device)
    target_hidden_tensor = torch.from_numpy(act_dict["target_hidden_state"][layer_index]).float().to(device)
    
    print(f"Processing hidden states for layer {layer_index} with mode '{mlp_mode}'")
    print(f"Draft hidden shape: {draft_hidden_tensor.shape}, Target hidden shape: {target_hidden_tensor.shape}")

    assert target_hidden_tensor.dim() == draft_hidden_tensor.dim(), \
        f"Draft and target hidden states must have the same dim. Got {draft_hidden_tensor.dim()} and {target_hidden_tensor.dim()}"
    
    if target_hidden_tensor.dim() == 2:
        draft_processed = draft_hidden_tensor
        target_processed = target_hidden_tensor
    elif target_hidden_tensor.dim() == 3:
        if mlp_mode == 'last_token':
            # Take the last token of every example
            draft_processed = draft_hidden_tensor[:, -1, :]
            target_processed = target_hidden_tensor[:, -1, :]
        elif mlp_mode == 'last_and_first_token':
            # Concatenate first & last token along hidden_dim
            draft_first = draft_hidden_tensor[:, 0, :]
            draft_last = draft_hidden_tensor[:, -1, :]
            draft_processed = torch.cat([draft_first, draft_last], dim=1)
            
            target_first = target_hidden_tensor[:, 0, :]
            target_last = target_hidden_tensor[:, -1, :]
            target_processed = torch.cat([target_first, target_last], dim=1)
        elif mlp_mode == 'all_tokens':
            # Flatten tokens into one vector
            n_t, s_t, h_t = target_hidden_tensor.shape
            n_d, s_d, h_d = draft_hidden_tensor.shape
            draft_processed = draft_hidden_tensor.reshape(n_d, s_d*h_d)
            target_processed = target_hidden_tensor.reshape(n_t, s_t*h_t)
            print(f"Draft processed shape: {draft_processed.shape}, Target processed shape: {target_processed.shape}")
        else:
            raise ValueError(f"Mode {mlp_mode!r} is not valid. Choose from: 'last_token', 'last_and_first_token', 'all_tokens'")
    else:
        raise ValueError(f"Target hidden state must be 2D or 3D. Got {target_hidden_tensor.dim()}D")

    print(f"Processed draft hidden shape: {draft_processed.shape}, target hidden shape: {target_processed.shape}")
    
    # Update the act_dict with processed tensors
    act_dict["draft_hidden_state"][layer_index] = draft_processed
    act_dict["target_hidden_state"][layer_index] = target_processed
    
    # Convert other fields to tensors
    act_dict["top_10_draft_logits"] = torch.from_numpy(act_dict["top_10_draft_logits"]).float().to(device)
    act_dict["top_10_target_logits"] = torch.from_numpy(act_dict["top_10_target_logits"]).float().to(device)
    act_dict["draft_entropies"] = torch.from_numpy(act_dict["draft_entropies"]).float().to(device).unsqueeze(-1)
    act_dict["target_entropies"] = torch.from_numpy(act_dict["target_entropies"]).float().to(device).unsqueeze(-1)
    act_dict["data_index"] = torch.from_numpy(act_dict["data_index"]).long().to(device).unsqueeze(-1)
    act_dict["step_index"] = torch.from_numpy(act_dict["step_index"]).long().to(device).unsqueeze(-1)

    # Ensure logits are 2D with shape [N, 1]
    draft_logits_tensor = torch.from_numpy(act_dict["draft_logits"]).float().to(device)
    target_logits_tensor = torch.from_numpy(act_dict["target_logits"]).float().to(device)
    
    if draft_logits_tensor.dim() == 1:
        draft_logits_tensor = draft_logits_tensor.unsqueeze(-1)
    if target_logits_tensor.dim() == 1:
        target_logits_tensor = target_logits_tensor.unsqueeze(-1)
    
    act_dict["draft_logits"] = draft_logits_tensor
    act_dict["target_logits"] = target_logits_tensor
    act_dict["selected_idxs"] = torch.from_numpy(act_dict["selected_idxs"]).long().to(device).unsqueeze(-1)
    act_dict["labels"] = torch.from_numpy(act_dict["labels"]).long().to(device)

    return act_dict

class AnchorDataset(Dataset):
    """Dataset that extracts features from activation dictionary for classifier training"""
    
    def __init__(self, act_dict, layer_index=-1):
        self.draft_hidden = act_dict["draft_hidden_state"][layer_index]
        self.target_hidden = act_dict["target_hidden_state"][layer_index]
        self.draft_top10 = act_dict["top_10_draft_logits"]
        self.target_top10 = act_dict["top_10_target_logits"]
        self.draft_entropy = act_dict["draft_entropies"]
        self.target_entropy = act_dict["target_entropies"]
        self.draft_logit = act_dict["draft_logits"]
        self.target_logit = act_dict["target_logits"]
        self.position = act_dict["selected_idxs"]
        self.labels = act_dict["labels"]
        self.data_index = act_dict["data_index"]
        self.step_index = act_dict["step_index"]
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "draft_hidden": self.draft_hidden[idx],
            "target_hidden": self.target_hidden[idx],
            "draft_top10": self.draft_top10[idx],
            "target_top10": self.target_top10[idx],
            "draft_entropy": self.draft_entropy[idx],
            "target_entropy": self.target_entropy[idx],
            "draft_logit": self.draft_logit[idx],
            "target_logit": self.target_logit[idx],
            "position": self.position[idx],
            "labels": self.labels[idx],
            "data_index": self.data_index[idx],
            "step_index": self.step_index[idx],
        }


def freeze_model(model):
    """Freeze all model parameters"""
    for param in model.parameters():
        param.requires_grad = False
    return model

def process_labels(labels):
    """Convert labels to binary format: False/0/-1 -> 0, True -> 1"""
    processed_labels = []
    for label in labels:
        if label is False or label in [0, -1]:
            processed_labels.append(0)
        elif label is True:
            processed_labels.append(1)
        else:
            raise ValueError(f"Label {label} is not valid")
    return processed_labels


def convert_single_token_sequence(row, source_tokenizer, target_tokenizer, source_model_name, train_data):
    """
    Convert a single token sequence from source model to target model.
    This function works on a single pandas row and is meant to be used with df.apply().
    
    Since only the last label is used downstream, we only preserve that.
    """
    tokens = row["tokens"]
    data_index = row["data_index"]
    
    # Find where model generation starts
    generation_start_idx = _find_generation_start(tokens, source_model_name)
    
    if generation_start_idx == -1:
        # If we can't find the generation start, return original
        raise ValueError("Generation start index not found in tokens.")
    
    # Extract the generated part
    generated_tokens = tokens[generation_start_idx:]
    
    # Decode the generated part
    generated_text = source_tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Recreate the prompt using the deterministic n-shot approach
    question = train_data[data_index]["question"]
    
    # Create 1-shot prompt (deterministic with seed=42)
    chats = nshot_chats(train_data, 1, question, seed=42)
    
    # Apply target model's chat template
    new_prompt_tokens = target_tokenizer.apply_chat_template(
        chats, 
        tokenize=True,
        add_generation_prompt=True
    )
    
    # Encode the new prompt
    # new_prompt_tokens = target_tokenizer.encode(prompt_text, add_special_tokens=True)
    
    # Encode the generated text
    new_generated_tokens = target_tokenizer.encode(generated_text, add_special_tokens=False)
    
    # Combine prompt and generation
    target_tokens = new_prompt_tokens + new_generated_tokens 

    # Target text
    # print(f"Target Text: {target_tokenizer.decode(target_tokens, skip_special_tokens=False)}")

    return target_tokens


def _find_generation_start(tokens, model_name):
    """
    Find where model generation starts by looking for model-specific tokens.
    Returns the index where generation starts, or -1 if not found.
    """
    if "qwen" in model_name.lower():
        # Look for </think> token (id 151668) for Qwen3
        think_end_token = 151668
        try:
            idx = tokens.index(think_end_token)
            return idx + 1  # Generation starts after </think>
        except ValueError:
            return -1
    
    elif "llama" in model_name.lower():
        # Look for the last occurrence of 128007 (end of assistant header)
        end_header_token = 128007
        try:
            idx = len(tokens) - 1 - tokens[::-1].index(end_header_token)
            return idx + 1  # Generation starts after the end header token
        except ValueError:
            return -1
    
    return -1


class MyTrainer:
    """Trainer class for collecting activations from target and draft models"""
    
    def __init__(self, target_model, draft_model, tokenizer):
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer

    def collect_activations(self, dataloader, layer_indices=[-1, -2, -3, -4, -5, -6, -7, -8], max_seq_len=1):
        """Collect activations from target and draft models for classifier training"""
        act_dict = {
            "draft_hidden_state": {k: [] for k in layer_indices},
            "target_hidden_state": {k: [] for k in layer_indices},
            "top_10_draft_logits": [],
            "top_10_target_logits": [],
            "draft_entropies": [],
            "target_entropies": [],
            "draft_logits": [],
            "target_logits": [],
            "selected_idxs": [],
            "labels": [],
            "prob_accepted": [],
            "prob_rejected": [],
            "data_index": [],
            "step_index": [],
        }

        self.target_model.eval()
        self.draft_model.eval()
        
        for batch in tqdm.tqdm(dataloader):
            with torch.no_grad():
                
                labels = batch["labels"].to(self.target_model.device)
                # We care about the last token label
                act_dict["labels"].extend(labels[:, -1].cpu().numpy())
                
                # Collect prob_accepted and prob_rejected
                prob_accepted = batch["prob_accepted"]
                prob_rejected = batch["prob_rejected"] 
                data_index = batch["data_index"]
                step_index = batch["step_index"]
                act_dict["prob_accepted"].extend(prob_accepted.numpy())
                act_dict["prob_rejected"].extend(prob_rejected.numpy())
                act_dict["data_index"].extend(data_index.numpy())
                act_dict["step_index"].extend(step_index.numpy())
                
                # Move batch to device
                input_ids = batch["input_ids"].to(self.target_model.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.target_model.device)


                # Target model
                target_logits, target_hidden_states = get_logits_and_embeddings(
                    self.target_model,
                    input_ids,
                    attention_mask=attention_mask,
                )

                # Draft model
                draft_logits, draft_hidden_states = get_logits_and_embeddings(
                    self.draft_model,
                    input_ids,
                    attention_mask=attention_mask,
                )

                # For each layer, get the hidden state of the last token
                for layer_index in layer_indices:
                    act_dict["target_hidden_state"][layer_index].extend(target_hidden_states[layer_index][:, -max_seq_len:, :].cpu().float().numpy())
                    act_dict["draft_hidden_state"][layer_index].extend(draft_hidden_states[layer_index][:, -max_seq_len:, :].cpu().float().numpy())

                # Get top 10 logits for draft and target
                draft_logits_for_calc = draft_logits[:, -2, :]
                target_logits_for_calc = target_logits[:, -2, :]
                draft_probs_for_calc = torch.softmax(draft_logits_for_calc, dim=-1)
                target_probs_for_calc = torch.softmax(target_logits_for_calc, dim=-1)
                
                # Get top 10 draft logits
                top_10_draft_logits = torch.topk(draft_probs_for_calc, 10, dim=-1).values
                act_dict["top_10_draft_logits"].extend(top_10_draft_logits.cpu().float().numpy())

                # Get top 10 target logits
                top_10_target_logits = torch.topk(target_probs_for_calc, 10, dim=-1).values
                act_dict["top_10_target_logits"].extend(top_10_target_logits.cpu().float().numpy())

                # Calculate entropies
                draft_entropies = -torch.sum(draft_probs_for_calc * torch.log(draft_probs_for_calc + 1e-10), dim=-1)
                act_dict["draft_entropies"].extend(draft_entropies.cpu().float().numpy())
                
                target_entropies = -torch.sum(target_probs_for_calc * torch.log(target_probs_for_calc + 1e-10), dim=-1)
                act_dict["target_entropies"].extend(target_entropies.cpu().float().numpy())

                # Get draft and target logits for the selected token
                selected_idxs = batch["input_ids"][:, -1].to(self.target_model.device)
                act_dict["selected_idxs"].extend(selected_idxs.cpu().numpy())
                draft_logits_for_token = draft_probs_for_calc.gather(1, selected_idxs.unsqueeze(-1))
                act_dict["draft_logits"].extend(draft_logits_for_token.cpu().float().numpy())
                target_logits_for_token = target_probs_for_calc.gather(1, selected_idxs.unsqueeze(-1))
                act_dict["target_logits"].extend(target_logits_for_token.cpu().float().numpy())
        return act_dict


def process_activations(act_dict, mode='last_token'):
    """
    Process activation lists into arrays with specified token aggregation mode.
    
    Args:
        act_dict: Dictionary of {layer: list of (seq_len, hidden_dim) arrays}
        mode: How to aggregate tokens - 'last_token', 'last_and_first_token', or 'all_tokens'
    
    Returns:
        new_act_dict: Dictionary of {layer: (n_examples, output_dim) arrays}
    """
    new_act_dict = {}
    for layer, activations in act_dict.items():
        print(f"Processing activations for layer {layer} with shape {activations[0].shape} and {len(activations)} examples")
        # Stack: shape = (n_examples, seq_len, hidden_dim)
        arr = np.stack(activations, axis=0)

        if mode == 'last_token':
            # Take the last token of every example
            new_act_dict[layer] = arr[:, -1, :]
        elif mode == 'last_and_first_token':
            # Concatenate first & last token along hidden_dim
            first = arr[:, 0, :]
            last = arr[:, -1, :]
            new_act_dict[layer] = np.concatenate([first, last], axis=1)
        elif mode == 'all_tokens':
            # Flatten tokens into one vector
            n, s, h = arr.shape
            new_act_dict[layer] = arr.reshape(n, s*h)
        else:
            raise ValueError(f"Mode {mode!r} is not valid")
        
        print(f"Processed activations for layer {layer} with shape {new_act_dict[layer].shape}")
    return new_act_dict


def prepare_data_as_df(data_base_dir, remove_duplicates=True, source_model_name=None, target_model_name=None, dataset="gsm8k"):
    """Load and prepare dataset as a pandas DataFrame"""
    print(f"Loading data from {data_base_dir}")
    data = []
    
    for file in os.listdir(data_base_dir):
        if file.startswith("dataset") or file.startswith("valid_dataset"):
            try:
                part = load_jsonlines(os.path.join(data_base_dir, file))
                data.extend(part)
            except Exception as e:
                print(f"Error in {file}: {e}")
                continue

    df = pd.DataFrame(data)
    print(f"Num Samples: {len(df)}")

    # Remove duplicates
    if remove_duplicates:
        if "index" in df.columns:
            df = df.drop_duplicates(subset=["data_index", "index"])
            print("Duplicates dropped")
        elif "step_index" in df.columns:
            df = df.drop_duplicates(subset=["data_index", "step_index"])
            print(f"Samples after dropping duplicates: {len(df)}")

    # Drop samples with soundness_check_failed=true
    if 'soundness_check_failed' in df.columns:
        df = df[df["soundness_check_failed"] != True]
        print("Samples with soundness_check_failed=true dropped")

    # Drop unnecessary columns, keeping only what's needed for analysis
    df = df.drop(
        columns=[
            "prompt",
            "generated",
            "accept_cont_label",
            "reject_cont_label",
            "reject_cont_tokens",
            "accept_cont_tokens",
            "verified",
            "soundness_check_failed",
            "accept_soundness",
            "skipped",
        ],
        errors="ignore",
    )

    print(f"Samples after preprocessing: {len(df)}")

    # Process labels
    df["label"] = df["label"].apply(lambda x: process_labels(x))

    # Convert tokens between models if needed
    if source_model_name is not None and target_model_name is not None and source_model_name != target_model_name:
        print(f"Converting tokens from {source_model_name} to {target_model_name}")

        # Check if data_index is required and available
        if "data_index" not in df.columns:
            print("Warning: data_index column not found, skipping token conversion")
        else:
            # Load tokenizers
            source_tokenizer = transformers.AutoTokenizer.from_pretrained(source_model_name)
            target_tokenizer = transformers.AutoTokenizer.from_pretrained(target_model_name)

            # Load training data for recreating prompts (only for gsm8k)
            if dataset == "gsm8k":
                train_data = load_jsonlines("data/gsm8k/train.jsonl")

                # Apply token conversion to each row
                print("Converting tokens using df.apply...")
                # Store original tokens for comparison
                original_tokens_example = df.iloc[0]["tokens"].copy() if len(df) > 0 else None

                df["tokens"] = df.apply(
                    lambda row: convert_single_token_sequence(
                        row, source_tokenizer, target_tokenizer, source_model_name, train_data
                    ), 
                    axis=1
                )
                print("Token conversion completed")

                # Print a single example of conversion
                if original_tokens_example is not None:
                    print(f"\nSINGLE CONVERSION EXAMPLE:")
                    original_text = source_tokenizer.decode(original_tokens_example, skip_special_tokens=False)
                    converted_text = target_tokenizer.decode(df.iloc[0]["tokens"], skip_special_tokens=False)
                    print(f"Original:  {original_text}...")
                    print(f"Converted: {converted_text}...")
                    print()
            else:
                print(f"Dataset {dataset} not supported for conversion yet")

    # Drop data_index after conversion (if not needed for training)
    # df = df.drop(columns=["data_index"], errors="ignore")

    # Ensure tokens and labels have the same length
    def align_lengths(row):
        if len(row['tokens']) > len(row['label']):
            extra_len = len(row['tokens']) - len(row['label'])
            row['label'] = [-100] * extra_len + row['label']
        elif len(row['tokens']) < len(row['label']):
            row['label'] = row['label'][-len(row['tokens']):]
        assert len(row['tokens']) == len(row['label'])
        return row
    df = df.apply(align_lengths, axis=1)

    # Truncate tokens to maximum length
    max_length = 10000
    df["tokens"] = df["tokens"].apply(lambda x: x[-max_length:])

    # Print label distribution
    num_ones = df["label"].apply(lambda x: x[-1]).sum()
    num_zeros = len(df) - num_ones
    print(f"Label distribution - Ones: {num_ones / len(df):.3f}, Zeros: {num_zeros / len(df):.3f}")
    
    return df


# Initialize configuration
args = Args()
torch.manual_seed(args.seed)
target_model_name = args.target_model_name
data_base_dir = args.data_base_dir
lr = args.lr
store_act = not args.load_act
layer_index = args.layer_index

# Setup directories
model_id = target_model_name.split("/")[-1]
save_name = f"sequential_extended_v10_{model_id}"

# Setup activation checkpoint directory
act_checkpoint_base_dir = f'{args.data_base_dir}/activations_{save_name}'
if args.identifier is not None:
    act_checkpoint_base_dir = f'{act_checkpoint_base_dir}_{args.identifier}'
if store_act:
    new_act_identifier = int(time.time())
    activation_save_dir = f'{act_checkpoint_base_dir}_{new_act_identifier}'
    os.makedirs(activation_save_dir, exist_ok=True)

# Setup model checkpoint directory
head_checkpoint_base_dir = f'{args.data_base_dir}/checkpoints_{save_name}'
if args.identifier is not None:
    head_checkpoint_base_dir = f'{head_checkpoint_base_dir}_{args.identifier}_{args.layer_index}_{args.mlp_mode}_{args.classifier_type}'
if args.save_head:
    new_save_head_identifier = int(time.time())
    head_checkpoint_base_dir = f'{head_checkpoint_base_dir}_{new_save_head_identifier}_{args.layer_index}_{args.mlp_mode}_{args.classifier_type}'
    os.makedirs(head_checkpoint_base_dir, exist_ok=True)

# Handle second data directory if provided
if args.data_base_dir_2 is not None:
    act_checkpoint_base_dir_2 = f'{args.data_base_dir_2}/activations_{save_name}'
    if args.identifier_2 is not None:
        act_checkpoint_base_dir_2 = f'{act_checkpoint_base_dir_2}_{args.identifier_2}'

# Save configuration
args_dict = vars(args)
if store_act:
    with open(os.path.join(activation_save_dir, "config.json"), "w") as f:
        json.dump(args_dict, f, indent=4)
if args.save_head:
    with open(os.path.join(head_checkpoint_base_dir, "config.json"), "w") as f:
        json.dump(args_dict, f, indent=4)

# Initialize tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(target_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
batch_size = args.batch_size

# Load or collect activations
if not args.load_act:
    df1 = prepare_data_as_df(
        data_base_dir, 
        remove_duplicates=args.remove_duplicates,
        source_model_name=args.source_model_name,
        target_model_name=args.target_model_name,
        dataset=args.dataset
    )
    if args.data_base_dir_2 is not None:
        df2 = prepare_data_as_df(
            args.data_base_dir_2, 
            remove_duplicates=args.remove_duplicates_db2,
            source_model_name=args.source_model_name,
            target_model_name=args.target_model_name,
            dataset=args.dataset
        )
        df = pd.concat([df1, df2], axis=0)
    else:
        df = df1


    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, padding=False
    )

    dataset = Dataset.from_pandas(df, preserve_index=False)
    dataset = dataset.map(
            lambda x: {
                "input_ids": x["tokens"], 
                "labels": x["label"], 
                "prob_accepted": x["prob_accepted"], 
                "prob_rejected": x["prob_rejected"],
                "data_index": x["data_index"],
                "step_index": x["step_index"]
            },
            remove_columns=["tokens", "label"],
        )
    dataloader = DataLoader(
        dataset, batch_size=1, collate_fn=data_collator, shuffle=False
    )

    print(f"Num Samples: {len(dataloader.dataset)}")
    print("Loading models...")

    # Setup quantization config if needed
    bnb_config = None
    if args.quantize:
        bnb_config = transformers.BitsAndBytesConfig(load_in_8bit=True)
        print(f"Using 8-bit quantization for {target_model_name}")

    # Load models
    causal_model_target = transformers.AutoModelForCausalLM.from_pretrained(
        target_model_name, 
        torch_dtype=torch.bfloat16, 
        device_map='auto', 
        quantization_config=bnb_config
    )
    
    causal_model_draft = transformers.AutoModelForCausalLM.from_pretrained(
        args.draft_model_name, 
        torch_dtype=torch.bfloat16, 
        device_map='auto'
    )

    target_model = freeze_model(causal_model_target)
    draft_model = freeze_model(causal_model_draft)

    # Collect activations
    trainer = MyTrainer(target_model, draft_model, tokenizer)
    layer_indices = args.collect_layer_indices
    act_dict = trainer.collect_activations(dataloader, layer_indices=layer_indices, max_seq_len=args.max_seq_len)

    # Convert lists of numpy arrays to single stacked numpy arrays for more efficient storage
    print("Converting activation lists to stacked numpy arrays...")
    for layer_index in layer_indices:
        print(f"Stacking activations for layer {layer_index}...")
        act_dict["target_hidden_state"][layer_index] = np.stack(act_dict["target_hidden_state"][layer_index], axis=0)
        act_dict["draft_hidden_state"][layer_index] = np.stack(act_dict["draft_hidden_state"][layer_index], axis=0)
    
    # Stack other arrays too
    act_dict["top_10_draft_logits"] = np.stack(act_dict["top_10_draft_logits"], axis=0)
    act_dict["top_10_target_logits"] = np.stack(act_dict["top_10_target_logits"], axis=0)
    act_dict["draft_entropies"] = np.array(act_dict["draft_entropies"])
    act_dict["target_entropies"] = np.array(act_dict["target_entropies"])
    act_dict["draft_logits"] = np.stack(act_dict["draft_logits"], axis=0)
    act_dict["target_logits"] = np.stack(act_dict["target_logits"], axis=0)
    act_dict["selected_idxs"] = np.array(act_dict["selected_idxs"])
    act_dict["labels"] = np.array(act_dict["labels"])
    act_dict["prob_accepted"] = np.array(act_dict["prob_accepted"])
    act_dict["prob_rejected"] = np.array(act_dict["prob_rejected"])
    act_dict["data_index"] = np.array(act_dict["data_index"])
    act_dict["step_index"] = np.array(act_dict["step_index"])
    print("Stacking completed!")

    with open(os.path.join(activation_save_dir, "act_dict.pkl"), "wb") as f:
        pickle.dump(act_dict, f)


layer_index = args.layer_index
print("\n" + "="*50)
print("PROB_ACCEPTED THRESHOLD ANALYSIS")
print("="*50)

print(f"Loading activations from {act_checkpoint_base_dir}")
if args.load_act:
    # Load the original act_dict to create a comprehensive dataframe
    with open(os.path.join(act_checkpoint_base_dir, "act_dict.pkl"), "rb") as f:
        original_act_dict = pickle.load(f)
else:
    original_act_dict = act_dict
print(f"Loaded activations")


# Create a comprehensive dataframe with all data
data_rows = []
# for i in range(len(original_act_dict["labels"])):
# use tqdm to show progress and operation
for i in tqdm.tqdm(range(len(original_act_dict["labels"])), desc="Creating DataFrame"):
    row = {
        'prob_accepted': original_act_dict["prob_accepted"][i],
        'prob_rejected': original_act_dict["prob_rejected"][i],
        'data_index': original_act_dict["data_index"][i],
        'step_index': original_act_dict["step_index"][i],
        'labels': original_act_dict["labels"][i],
        'draft_hidden': original_act_dict["draft_hidden_state"][layer_index][i],
        'target_hidden': original_act_dict["target_hidden_state"][layer_index][i],
        'draft_top10': original_act_dict["top_10_draft_logits"][i],
        'target_top10': original_act_dict["top_10_target_logits"][i],
        'draft_entropy': original_act_dict["draft_entropies"][i],
        'target_entropy': original_act_dict["target_entropies"][i],
        'draft_logit': original_act_dict["draft_logits"][i],
        'target_logit': original_act_dict["target_logits"][i],
        'selected_idx': original_act_dict["selected_idxs"][i]
    }
    data_rows.append(row)

# Create dataframe
full_df = pd.DataFrame(data_rows)

# Test different prob_accepted thresholds
prob_accepted_thresholds = [1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.0]
classifier_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
threshold_results = []


# Get base hidden dimensions from the first sample (which is a seq_len x hidden_dim array)
sample_target_hidden = torch.tensor(full_df['target_hidden'].iloc[0])
sample_draft_hidden = torch.tensor(full_df['draft_hidden'].iloc[0])

seq_len, base_target_hidden_dim = sample_target_hidden.shape
_, base_draft_hidden_dim = sample_draft_hidden.shape

# Calculate effective dimensions based on mlp_mode
if args.mlp_mode == 'last_token':
    target_hidden_dim = base_target_hidden_dim
    draft_hidden_dim = base_draft_hidden_dim
elif args.mlp_mode == 'last_and_first_token':
    target_hidden_dim = base_target_hidden_dim * 2
    draft_hidden_dim = base_draft_hidden_dim * 2
elif args.mlp_mode == 'all_tokens':
    target_hidden_dim = base_target_hidden_dim * seq_len
    draft_hidden_dim = base_draft_hidden_dim * seq_len
else:
    raise ValueError(f"Invalid mlp_mode: {args.mlp_mode}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Sample hidden shape: {sample_target_hidden.shape} (seq_len x hidden_dim)")
print(f"Base target hidden dim: {base_target_hidden_dim}, Base draft hidden dim: {base_draft_hidden_dim}")
print(f"Effective target hidden dim: {target_hidden_dim}, Effective draft hidden dim: {draft_hidden_dim} (mode: {args.mlp_mode})")

def df_to_act_dict(df, layer_index=-1):
    """Convert DataFrame back to activation dictionary format"""
    return {
        "draft_hidden_state": {layer_index: np.stack(df['draft_hidden'].values, axis=0)},
        "target_hidden_state": {layer_index: np.stack(df['target_hidden'].values, axis=0)},
        "top_10_draft_logits": np.stack(df['draft_top10'].values, axis=0),
        "top_10_target_logits": np.stack(df['target_top10'].values, axis=0),
        "draft_entropies": np.array(df['draft_entropy'].values),
        "target_entropies": np.array(df['target_entropy'].values),
        "draft_logits": np.stack(df['draft_logit'].values, axis=0),
        "target_logits": np.stack(df['target_logit'].values, axis=0),
        "selected_idxs": np.array(df['selected_idx'].values),
        "labels": np.array(df['labels'].values),
        "prob_accepted": np.array(df['prob_accepted'].values),
        "prob_rejected": np.array(df['prob_rejected'].values),
        "data_index": np.array(df['data_index'].values),
        "step_index": np.array(df['step_index'].values),
    }

for threshold in prob_accepted_thresholds:
    print(f"\n{'='*30}")
    print(f"Training with prob_accepted <= {threshold}")
    print(f"{'='*30}")

    print(f"Initial samples: {len(full_df)}")
    # Remove samples where prob_rejected is zero
    working_df = full_df[full_df['prob_rejected'] > 0].copy()
    print(f"Samples after removing prob_rejected == 0: {len(working_df)}")

    print("Samples where target logit is zero and label is 0:")
    zero_logit_samples = full_df[(full_df['target_logit'] == 0) & (full_df['labels'] == 0)]
    print(zero_logit_samples)

    print("Samples where label is 1 and prob_accepted is greater than threshold:")
    one_accepted_samples = full_df[(full_df['labels'] == 1) & (full_df['prob_accepted'] > threshold)]
    print(one_accepted_samples)

    # Filter dataframe based on prob_accepted threshold
    # filtered_df = working_df[(working_df['prob_accepted'] <= threshold) | (working_df['prob_accepted'] == 1)].copy()
    print(f"Total samples: {len(full_df)}")
    print(f"prob_accepted range: [{full_df['prob_accepted'].min():.3f}, {full_df['prob_accepted'].max():.3f}]")

    # Filter dataframe to only keep samples where prob_accepted is less than or equal to the threshold or label is 0 and prob_accepted is equal to prob_rejected
    # filtered_df = working_df[(working_df['prob_accepted'] <= threshold) | (working_df['labels'] == 0) & (working_df['prob_accepted'] == working_df['prob_rejected'])].copy()

    filtered_df = working_df[
        (  (working_df["labels"] == 1) & (working_df["prob_accepted"] <= threshold))
        | ((working_df["labels"] == 0) & (working_df["prob_accepted"] >= 0.8))
    ].copy()


    print(f"Filtered samples: {len(filtered_df)}")

    # Calculate class distribution
    num_positive = (filtered_df['labels'] == 1).sum()
    num_negative = (filtered_df['labels'] == 0).sum()
    positive_ratio = num_positive / len(filtered_df)

    print(f"Positive samples: {num_positive} ({positive_ratio:.3f})")
    print(f"Negative samples: {num_negative} ({1-positive_ratio:.3f})")

    # Convert filtered dataframe to act_dict and then to tensors
    filtered_act_dict = df_to_act_dict(filtered_df, layer_index=layer_index)
    filtered_act_dict = convert_act_dict_to_tensors(filtered_act_dict, layer_index=layer_index, mlp_mode=args.mlp_mode, device=device)

    # Create dataset
    filtered_dataset = AnchorDataset(filtered_act_dict, layer_index=layer_index)

    # Split into train/test
    filtered_train_dataset, filtered_test_dataset = torch.utils.data.random_split(
        filtered_dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
    )

    filtered_train_loader = torch.utils.data.DataLoader(
        filtered_train_dataset, batch_size=batch_size, shuffle=True
    )
    filtered_test_loader = torch.utils.data.DataLoader(
        filtered_test_dataset, batch_size=batch_size, shuffle=False
    )


    # Create and train model for this threshold
    torch.manual_seed(42)
    # filtered_model = AnchorClassifierExtendedTarget(
    #     draft_hidden_dim=draft_hidden_dim,
    #     target_hidden_dim=target_hidden_dim
    # ).to(device)
    output_dim = 2
    if args.classifier_type == 'linear':
        hidden_dim = []
        filtered_model = TorchMLP(target_hidden_dim, hidden_dim, output_dim).to(device)
    elif args.classifier_type == 'mlp':
        hidden_dim = [400, 200]
        filtered_model = TorchMLP(target_hidden_dim, hidden_dim, output_dim).to(device)
    elif args.classifier_type == 'extended':
        filtered_model = AnchorClassifierExtendedTarget(
            target_hidden_dim=target_hidden_dim,
            hidden_dim=128,
            t_embed=64, 
            s_embed=64,
        ).to(device)
    elif args.classifier_type == 'extended_v2':
        filtered_model = AnchorClassifierExtendedTargetV2(
            target_hidden_dim=target_hidden_dim,
            hidden_dim=128,
            t_embed=64, 
            s_embed=16,
        ).to(device)
    else:
        raise ValueError("Invalid classifier type. Choose 'mlp' or 'linear'.")
    filtered_model.eval()


    filtered_optimizer = torch.optim.AdamW(filtered_model.parameters(), lr=lr)

    filtered_class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(filtered_df['labels']),
        y=filtered_df['labels'].values
    )
    filtered_class_weights = torch.tensor(filtered_class_weights).float().to(device)

    print(f"Filtered class weights: {filtered_class_weights}")

    filtered_criterion = nn.CrossEntropyLoss(weight=filtered_class_weights)

    # Training loop - reduced epochs for faster analysis
    n_epochs_filtered = args.n_epochs

    print(f"Training for {n_epochs_filtered} epochs...")
    print(f"Caching {len(filtered_train_loader)} training batches and {len(filtered_test_loader)} test batches...")
    filtered_batches = [batch for batch in filtered_train_loader]
    filtered_test_batches = [batch for batch in filtered_test_loader]
    print(f"Cached {len(filtered_batches)} training batches and {len(filtered_test_batches)} test batches.")
    best_filtered_test_loss = float('inf')
    best_filtered_model_state = None
    for epoch in range(n_epochs_filtered):
        for batch in filtered_batches:
        # for batch in filtered_train_loader:
            filtered_model.train()
            filtered_optimizer.zero_grad()

            for k, v in batch.items():
                batch[k] = v.to(device)

            if "labels" in batch:
                batch_labels = batch.pop("labels")

            outputs = filtered_model(**batch)
            loss = filtered_criterion(outputs, batch_labels)
            loss.backward()
            filtered_optimizer.step()

        if epoch % 10 == 0 or epoch == n_epochs_filtered - 1:
            metrics = eval(filtered_model, filtered_test_batches, [0.5], loss_fn=filtered_criterion)
            test_loss = metrics[0.5]['test_loss']
            print(
                f"Step {epoch}/{n_epochs_filtered}, Loss: {loss.item()}, Test Loss: {test_loss}, F1: {metrics[0.5]['f1']}, Precision: {metrics[0.5]['precision']}, Recall: {metrics[0.5]['recall']}, Accuracy: {metrics[0.5]['accuracy']}"
            )
            if test_loss < best_filtered_test_loss:
                best_filtered_test_loss = test_loss
                best_filtered_model_state = copy.deepcopy(filtered_model.state_dict())
                print(f"New best model found with test loss: {best_filtered_test_loss}")

    # Load best model state if available
    if best_filtered_model_state:
        filtered_model.load_state_dict(best_filtered_model_state)
        print("Loaded best model with lowest test loss for evaluation.")
        
    # Evaluate the filtered model across different classifier thresholds
    filtered_metrics_dict = eval(filtered_model, filtered_test_loader, classifier_thresholds)

    # Get predictions for ROC curve plotting
    filtered_model.eval()
    all_test_probs = []
    all_test_labels = []
    with torch.no_grad():
        for batch in filtered_test_loader:
            for k, v in batch.items():
                batch[k] = v.to(device)
            labels = batch.pop("labels")
            batch.pop("data_index", None)
            batch.pop("step_index", None)
            outputs = filtered_model(**batch)
            output_proba = torch.softmax(outputs, dim=1)[:, 1]
            all_test_probs.extend(output_proba.cpu().numpy())
            all_test_labels.extend(labels.cpu().numpy())
    
    all_test_probs = np.array(all_test_probs)
    all_test_labels = np.array(all_test_labels)

    if args.save_head:
        # Save the model head for this threshold
        head_save_path = os.path.join(head_checkpoint_base_dir, f"anchor_classifier_head_{args.classifier_type}_layer{layer_index}_epochs{n_epochs_filtered}_{threshold:.1f}_f1{filtered_metrics_dict[0.5]['f1']:.3f}_recall{filtered_metrics_dict[0.5]['recall']:.3f}.pt")
        torch.save(filtered_model.state_dict(), head_save_path)
        print(f"Saved model head for threshold {threshold} to {head_save_path}")
        
        # Save predictions for ROC curve plotting
        predictions_save_path = os.path.join(head_checkpoint_base_dir, f"test_predictions_{threshold:.1f}.npz")
        np.savez(predictions_save_path, 
                 probabilities=all_test_probs, 
                 labels=all_test_labels)
        print(f"Saved test predictions for threshold {threshold} to {predictions_save_path}")


    print(f"\nResults for prob_accepted <= {threshold}:")
    print(f"{'Classifier Threshold':<20} {'F1':<8} {'Precision':<12} {'Recall':<8} {'Accuracy':<10}")
    print("-" * 60)

    for clf_threshold in classifier_thresholds:
        metrics = filtered_metrics_dict[clf_threshold]
        print(f"{clf_threshold:<20.1f} {metrics['f1']:<8.3f} {metrics['precision']:<12.3f} {metrics['recall']:<8.3f} {metrics['accuracy']:<10.3f}")
        # print(f"Wrong Data Indices: {metrics.get('wrong_data_index', [])}")
        # print(f"Wrong Step Indices: {metrics.get('wrong_step_index', [])}")
    print("-" * 60)

    # Store results for the 0.5 threshold for overall comparison
    threshold_results.append({
        'threshold': threshold,
        'num_samples': len(filtered_df),
        'num_positive': num_positive,
        'num_negative': num_negative,
        'positive_ratio': positive_ratio,
        'f1': filtered_metrics_dict[0.5]['f1'],
        'precision': filtered_metrics_dict[0.5]['precision'],
        'recall': filtered_metrics_dict[0.5]['recall'],
        'accuracy': filtered_metrics_dict[0.5]['accuracy'],
        'all_metrics': filtered_metrics_dict
    })

    # Clean up memory
    del filtered_df, filtered_act_dict, filtered_dataset, filtered_test_dataset
    del filtered_train_dataset
    del filtered_model
    del filtered_optimizer
    torch.cuda.empty_cache()