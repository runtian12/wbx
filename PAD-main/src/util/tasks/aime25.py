import re
from datasets import load_dataset
from src.util.tasks.base import BaseTaskUtils
from src.util.general import get_text
from math_verify import parse, verify, math_metric
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig

verification_metric = math_metric(
    gold_extraction_target=(LatexExtractionConfig(boxed_match_priority=0),),
    pred_extraction_target=(
        LatexExtractionConfig(),
        ExprExtractionConfig(),
    ),
    aggregation_function=max,
    precision=6
)

class AIME25TaskUtils(BaseTaskUtils):
    """
    Utility class for handling AIME 2025 dataset tasks.
    This class provides methods to load the dataset from HuggingFace,
    evaluate answers, and prepare samples.
    """
    @staticmethod
    def get_task_name():
        return "aime25"

    @staticmethod
    def load_dataset(base_path=None):
        # Load AIME25 dataset from HuggingFace
        dataset = load_dataset("math-ai/aime25", split="train")
        
        # Put everything in test set as requested
        train_data = []
        test_data = [
            {
                "problem": item["problem"],
                "answer": item["answer"],
                "id": idx
            }
            for idx, item in enumerate(dataset)
        ]
        
        return train_data, test_data
    
    @staticmethod
    def polish_ground_truth(txt):
        return txt.strip()

    @staticmethod
    def evaluate(sequence_or_text, reference_txt, sample, start_idx=0, tokenizer=None, polish_reference=True):
        sequence_txt = get_text(sequence_or_text, tokenizer, start_idx=start_idx, skip_special_tokens=True)
        
        # Get the gold answer
        if polish_reference:
            gold_answer = AIME25TaskUtils.polish_ground_truth(sample["answer"])
        else:
            gold_answer = sample["answer"]
        
        # Format gold answer for verification
        gold_answer = f"\\boxed{{{gold_answer}}}"

        score, extracted_info = verification_metric([gold_answer], [sequence_txt])
        is_correct = score == 1.0
        
        return is_correct

    @staticmethod
    def prepare_sample(
        train_data, data_index, tokenizer, n_shot=0, enable_thinking=False
    ):
        # Since all data is in test_data, this would typically be used with test_data
        qna = train_data[data_index] if train_data else {}
        
        ground_truth = qna.get("answer", "")
        problem = qna.get("problem", "")
        
        prompt = f"""Question: {problem}

Please solve this step by step and provide your final answer in \\boxed{{}} format.

Step-by-Step Answer:"""
        
        messages = [{"role": "user", "content": prompt}]
        current_sequence = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt",
            enable_thinking=enable_thinking
        )

        return current_sequence, ground_truth, prompt, qna

    @staticmethod
    def get_question(sample):
        return sample["problem"]
        
    @staticmethod
    def polish_answer(answer):
        return answer.strip()

    @staticmethod
    def check_soundness(sample, response_text):
        raise NotImplementedError("Soundness check not implemented for AIME25.")