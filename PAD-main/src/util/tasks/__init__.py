
from .gsm8k import GSM8kTaskUtils
from .math import MATHTaskUtils
from .base import BaseTaskUtils
from .mbpp import MBPPTaskUtils
from .apps import APPSTaskUtils
from .aime24 import AIME24TaskUtils
from .aime25 import AIME25TaskUtils
from .trivia_qa import TriviaQATaskUtils


class TaskManager:
    """
    Manager class to handle different task utilities.
    This class provides a unified interface to access task-specific utility functions.
    """

    @staticmethod
    def get_task_utils(task_name):
        task_name = task_name.lower()
        if task_name == "gsm8k":
            return GSM8kTaskUtils()
        elif task_name == "math":
            return MATHTaskUtils()
        elif task_name == "mbpp":
            return MBPPTaskUtils()
        elif task_name == "apps":
            return APPSTaskUtils()
        elif task_name == "trivia_qa":
            return TriviaQATaskUtils()
        elif task_name == "aime24":
            return AIME24TaskUtils()
        elif task_name == "aime25":
            return AIME25TaskUtils()
        else:
            raise ValueError(f"Unknown task name: {task_name}")

    @staticmethod
    def get_all_task_names():
        return ["gsm8k", "math", "mbpp", "apps", "trivia_qa", "aime24", "aime25"]

    
    @staticmethod
    def load_dataset(task_name, base_path=None):
        task_utils = TaskManager.get_task_utils(task_name)
        return task_utils.load_dataset(base_path=base_path)

    @staticmethod
    def get_task_name(task_name):
        task_utils = TaskManager.get_task_utils(task_name)
        return task_utils.get_task_name()

    @staticmethod
    def polish_ground_truth(task_name, txt):
        task_utils = TaskManager.get_task_utils(task_name)
        return task_utils.polish_ground_truth(txt)

    @staticmethod
    def evaluate(task_name, sequence_or_text, reference_txt, sample, start_idx=0, tokenizer=None, polish_reference=True):
        task_utils = TaskManager.get_task_utils(task_name)
        return task_utils.evaluate(sequence_or_text, reference_txt, sample, start_idx, tokenizer, polish_reference)

    @staticmethod
    def prepare_sample(task_name, train_data, data_index, tokenizer, enable_thinking=False):
        task_utils = TaskManager.get_task_utils(task_name)
        return task_utils.prepare_sample(train_data, data_index, tokenizer, enable_thinking=enable_thinking)

    @staticmethod
    def get_question(task_name, sample):
        task_utils = TaskManager.get_task_utils(task_name)
        return task_utils.get_question(sample)

    @staticmethod
    def polish_answer(task_name, answer):
        task_utils = TaskManager.get_task_utils(task_name)
        return task_utils.polish_answer(answer)

    @staticmethod
    def check_soundness(task_name, sample, response_text):
        task_utils = TaskManager.get_task_utils(task_name)
        return task_utils.check_soundness(sample, response_text)



