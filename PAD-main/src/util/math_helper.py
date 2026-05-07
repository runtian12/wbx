import random
import re
import sys
import os

# Add the deepseek-evaluation directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
deepseek_eval_path = os.path.join(current_dir, '..', '..', 'deepseek-evaluation')
deepseek_eval_path = os.path.abspath(deepseek_eval_path)
if deepseek_eval_path not in sys.path:
    sys.path.append(deepseek_eval_path)

# Import DeepSeek's answer extraction and evaluation functions
try:
    from data_processing.answer_extraction import extract_answer as deepseek_extract_answer, extract_math_answer
    from eval.eval_utils import math_equal as deepseek_compare_answers
    DEEPSEEK_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import DeepSeek functions: {e}")
    DEEPSEEK_AVAILABLE = False


def nshot_chats(nshot_data: list, n: int, problem: str, seed: int = None):
    """
    Create few-shot chat format for MATH dataset problems using DeepSeek's approach.
    
    Args:
        nshot_data: List of MATH dataset examples
        n: Number of examples to include
        problem: The problem statement to solve
        seed: Random seed for example selection
    
    Returns:
        List of chat messages formatted for the model
    """
    chats = []
    
    # Set random seed
    random.seed(42 if seed is None else seed)
    
    # Sample n examples for few-shot learning
    for example in random.sample(nshot_data, min(n, len(nshot_data))):
        chats.append(
            {"role": "user", "content": example["problem"]}
        )
        chats.append(
            {"role": "assistant", "content": example["solution"]}
        )

    # Add the current problem to solve with DeepSeek's prompt format
    chats.append({
        "role": "user", 
        "content": f"{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    })

    return chats


def extract_answer_from_response(response: str, eos=None):
    """
    Extract the final answer from a MATH dataset response using DeepSeek's method.
    
    Args:
        response: The model's response containing the solution
        eos: End-of-sequence token to truncate at
    
    Returns:
        The extracted answer or empty string if extraction fails
    """
    if eos:
        response = response.split(eos)[0].strip()
    
    if DEEPSEEK_AVAILABLE:
        # Use DeepSeek's extract_answer function
        try:
            answer = deepseek_extract_answer(response)
            return answer if answer else ""
        except Exception as e:
            print(f"DeepSeek extract_answer failed: {e}")
    
    # Fallback implementation
    # Look for \\boxed{} pattern which is the standard format for MATH answers
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(boxed_pattern, response)
    
    if matches:
        answer = matches[-1].strip()  # Take the last match
        return answer
    
    # Fallback: look for patterns like "The answer is X" or "Therefore, X"
    answer_patterns = [
        r'[Tt]he answer is\s*([^.]+)',
        r'[Tt]herefore,?\s*([^.]+)',
        r'[Ss]o,?\s*([^.]+)',
        r'[Hh]ence,?\s*([^.]+)',
    ]
    
    for pattern in answer_patterns:
        matches = re.findall(pattern, response)
        if matches:
            answer = matches[-1].strip()
            return answer
    
    return ""


def normalize_answer(answer: str):
    """
    Normalize mathematical expressions for comparison using DeepSeek's approach.
    
    Args:
        answer: The answer string to normalize
    
    Returns:
        Normalized answer string
    """
    if not answer:
        return ""
    
    # Remove common mathematical formatting
    answer = answer.replace('$', '')  # Remove dollar signs
    answer = answer.replace('\\', '')  # Remove backslashes
    answer = answer.replace('{', '').replace('}', '')  # Remove braces
    answer = answer.replace(' ', '')  # Remove spaces
    
    # Handle common fraction formats
    if '/' in answer:
        try:
            # Try to evaluate simple fractions
            parts = answer.split('/')
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                numerator = int(parts[0])
                denominator = int(parts[1])
                # Return as simplified fraction or decimal
                if denominator != 0:
                    from fractions import Fraction
                    frac = Fraction(numerator, denominator)
                    return str(frac)
        except:
            pass
    
    # Handle decimal numbers
    try:
        # Try to convert to float and back to handle different decimal representations
        float_val = float(answer)
        # If it's a whole number, return as integer
        if float_val.is_integer():
            return str(int(float_val))
        else:
            return str(float_val)
    except:
        pass
    
    # Return as-is if no normalization was possible
    return answer.strip()


def compare_answers(predicted: str, ground_truth: str):
    """
    Compare predicted answer with ground truth for MATH dataset using DeepSeek's method.
    
    Args:
        predicted: The predicted answer
        ground_truth: The ground truth answer
    
    Returns:
        Boolean indicating if answers match
    """
    if not predicted or not ground_truth:
        return False
    
    if DEEPSEEK_AVAILABLE:
        # Use DeepSeek's math_equal function
        try:
            return deepseek_compare_answers(predicted, ground_truth)
        except Exception as e:
            print(f"DeepSeek compare_answers failed: {e}")
    
    # Fallback implementation
    # Normalize both answers
    norm_pred = normalize_answer(predicted)
    norm_gt = normalize_answer(ground_truth)
    
    # Direct string comparison
    if norm_pred == norm_gt:
        return True
    
    # Try numerical comparison
    try:
        pred_num = float(norm_pred)
        gt_num = float(norm_gt)
        # Use a small epsilon for floating point comparison
        return abs(pred_num - gt_num) < 1e-6
    except:
        pass
    
    # Try fraction comparison
    try:
        from fractions import Fraction
        pred_frac = Fraction(norm_pred)
        gt_frac = Fraction(norm_gt)
        return pred_frac == gt_frac
    except:
        pass
    
    return False
