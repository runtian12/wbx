import re
from src.util.json_io import load_jsonlines
from src.util.tasks.base import BaseTaskUtils
from src.util.gsm8k_helper import nshot_chats
from src.util.general import get_text
from math_verify import parse, verify, math_metric
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
from src.util.tasks.gsm8k import soundness_check_prompt_template

verification_metric = math_metric(
    gold_extraction_target=(LatexExtractionConfig(boxed_match_priority=0),),
    pred_extraction_target=(
        LatexExtractionConfig(),
        ExprExtractionConfig(),
    ),
    aggregation_function=max,
    precision=6
)

class MATHTaskUtils(BaseTaskUtils):
    """
    Utility class for handling math dataset tasks.
    This class provides methods to load the dataset, polish references,
    evaluate answers, and prepare samples.
    """
    @staticmethod
    def get_task_name():
        return "math"

    @staticmethod
    def load_dataset(base_path="data/math_splits"):
        train_data = load_jsonlines(f"{base_path}/train.jsonl")
        test_data = load_jsonlines(f"{base_path}/test.jsonl")
        return train_data, test_data
    
    @staticmethod
    def polish_ground_truth(txt):
        return txt.strip()

    @staticmethod
    def evaluate(sequence_or_text, reference_txt, sample, start_idx=0, tokenizer=None, polish_reference=True):
        sequence_txt = get_text(sequence_or_text, tokenizer, start_idx=start_idx, skip_special_tokens=True)
        gold_answer = sample["answer"]

        # put the gold answer in latex format
        gold_answer = f"${gold_answer}$"

        score, extracted_info = verification_metric([gold_answer], [sequence_txt])
        is_correct = score == 1.0
        
        # if extracted_info is not None:
        #     extracted_gold, extracted_pred = extracted_info
        # else:
        #     extracted_gold, extracted_pred = None, None

        # print(f"Gold Answer: {gold_answer}")
        # print(f"Extracted Gold: {extracted_gold}")
        # print(f"Extracted Pred: {extracted_pred}")
        # print(f"Is Correct: {is_correct}")
        return is_correct

    @staticmethod
    def prepare_sample(
        train_data, data_index, tokenizer, n_shot=1,  enable_thinking=False
    ):
        qna = train_data[data_index]
        # keys: problem, solution, answer, subject, level, unique_id
        ground_truth = qna["answer"]
        problem = qna["problem"]
        prompt = f"""Question: {problem}

Please solve this step by step and provide your final answer in \\boxed{{}} format.

Step-by-Step Answer:"""
        messages = [{"role": "user", "content": prompt}]
        current_sequence = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt",
            enable_thinking=enable_thinking
        )

        print(f"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print(f"Problem: {problem}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Prompt: {prompt}")
        print(f"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        return current_sequence, ground_truth, prompt, qna


    @staticmethod
    def get_question(sample):
        return sample["problem"]
        
    @staticmethod
    def polish_answer(answer):
        return answer.strip()


    @staticmethod
    def check_soundness(sample, response_text):
        final_prompt = soundness_check_prompt_template.format(
            question=sample["problem"],
            ground_truth=sample["solution"],
            response=response_text
        )

        return MATHTaskUtils.call_api(
            prompt=final_prompt
        )