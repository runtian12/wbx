from torch import Tensor
def get_text(sequence_or_text, tokenizer, start_idx=0, skip_special_tokens=False):
    """
    Get the text from a sequence tensor starting from start_idx.
    """
    if isinstance(sequence_or_text, Tensor):
        return tokenizer.decode(sequence_or_text[start_idx:], skip_special_tokens=skip_special_tokens)
    elif isinstance(sequence_or_text, str):
        return sequence_or_text  # Assuming sequence is already a string
    else:
        raise ValueError("Input must be a tensor or a string.")