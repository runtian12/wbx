import re
from src.util.json_io import load_jsonlines
from src.util.tasks.base import BaseTaskUtils
from src.util.gsm8k_helper import nshot_chats
from src.util.general import get_text
import random
from math_verify import parse, verify, math_metric
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig

soundness_check_prompt_template = """
You are an expert evaluator specializing in identifying flawed reasoning in problem-solving narratives. Your task is to analyze a given response and determine if it contains logical errors, even if the final answer is correct.

Read the entire 'Response to Analyze' from start to finish. A response is considered to have flawed reasoning if you identify any of the following patterns:

- **Incorrect Intermediate Steps:** It contains calculation errors or incorrect logical steps that are later implicitly corrected or ignored to reach the correct final answer.
- **Logical Fallacies:** The reasoning contains leaps of logic or doesn't coherently lead to the conclusion, but the result coincidentally matches the correct answer.
- **Self-Correction:** The response states an incorrect piece of information or result and then corrects itself. Sometimes (not always) there are explicit phrases like "Wait, that's not right," "But the question asks for...," or "Let's try again" before the correct path is found.
- **Unnecessary or Redundant Steps:** It includes steps or calculations that are irrelevant and do not contribute to the final answer, indicating a confused or inefficient thought process.
- **Other Reasoning Flaws:** Any other type of faulty logic that raises questions about the soundness of the process.

---

### **Examples**

**EXAMPLE 1: FLAWED REASONING (Self-Correction)**

* **Question:** A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?
* **Response to Analyze:** The bat costs $1.00, so the ball must cost $0.10. Wait, that's not right because the difference is only $0.90. Let's try again. If the ball is B, the bat is B + 1. So B + (B + 1) = 1.10, which means 2B = 0.10. The ball costs $0.05.
* **Your JSON Output:**
    ```json
    {{
      "analysis": "The response initially presents an incorrect answer ($0.10) but then immediately identifies the error and uses a correct algebraic method to perform a self-correction, arriving at the right answer.",
      "decision": true
    }}
    ```

**EXAMPLE 2: SOUND REASONING (No Flaws)**

* **Question:** A bakery has 5 boxes of donuts, with 12 donuts in each box. They sell 3 boxes. How many donuts are left?
* **Response to Analyze:** First, determine the number of boxes remaining, which is 5 - 3 = 2 boxes. Then, calculate the total donuts in the remaining boxes: 2 boxes * 12 donuts/box = 24 donuts.
* **Your JSON Output:**
    ```json
    {{
      "analysis": "The reasoning is sound, direct, and efficient. It correctly calculates the remaining boxes before finding the total number of donuts left. There are no logical flaws.",
      "decision": false
    }}
    ```

**EXAMPLE 3: FLAWED REASONING (Unnecessary Steps)**

* **Question:** A recipe requires 2 cups of flour to make 12 cookies. You want to make 36 cookies. How much flour do you need?
* **Response to Analyze:** To make 36 cookies, which is 3 times 12, you'll need 3 times the flour. So, 2 cups * 3 = 6 cups. The oven should be preheated to 350°F. The total flour needed is 6 cups.
* **Your JSON Output:**
    ```json
    {{
      "analysis": "The core calculation is correct, but the response introduces an unnecessary and irrelevant piece of information ('The oven should be preheated to 350°F') that does not contribute to solving the problem.",
      "decision": true
    }}
    ```

---

### **Input Data for Analysis**

**1. Question:**
{question}

**2. Golden Answer (for your reference):**
{ground_truth}

**3. Response to Analyze:**
{response}

---

### **Your Task**

After carefully reviewing the 'Response to Analyze', provide your evaluation in a strict JSON format, following the structure shown in the examples above. The JSON object must contain exactly two keys: `"analysis"` and `"decision"`.

**Your JSON Output:**
"""


# After carefully reviewing the 'Response to Analyze', provide your evaluation in a strict JSON format, following the structure shown in the examples above. The JSON object must be single line and contain exactly two keys: `"analysis"` (string) and `"decision"` (boolean). In `"analysis"`: plain ASCII only; no backslashes, LaTeX, dollar signs, backticks, or internal double quotes; no newlines. E.g. write math as 3^6 mod 17. Output nothing else.




verification_metric = math_metric(
    gold_extraction_target=(LatexExtractionConfig(boxed_match_priority=0),),
    pred_extraction_target=(
        LatexExtractionConfig(),
        ExprExtractionConfig(),
    ),
    aggregation_function=max,
    precision=6
)


class GSM8kTaskUtils(BaseTaskUtils):
    """
    Utility class for handling GSM8k dataset tasks.
    This class provides methods to load the dataset, polish references,
    evaluate answers, and prepare samples.
    """
    @staticmethod
    def get_task_name():
        return "gsm8k"
    
    @staticmethod
    def load_dataset(base_path="data/gsm8k"):
        train_data = load_jsonlines(f"{base_path}/train.jsonl")
        test_data = load_jsonlines(f"{base_path}/test.jsonl")
        # Select 500 randomly with seed 42
        rng = random.Random(42)
        test_data = rng.sample(test_data, k=200)
        return train_data, test_data
    
    @staticmethod
    def polish_ground_truth(txt):
        reference_ans = txt.split("####")[-1].strip()
        return GSM8kTaskUtils.polish_answer(reference_ans)

    # @staticmethod
    # def evaluate(sequence_or_text, reference_txt, sample, start_idx=0, tokenizer=None, polish_reference=True):
    #     if polish_reference:
    #         reference_ans = GSM8kTaskUtils.polish_ground_truth(reference_txt)
    #     else:
    #         reference_ans = reference_txt



    #     sequence_txt = get_text(sequence_or_text, tokenizer, start_idx=start_idx, skip_special_tokens=True)
    #     seq_answer = sequence_txt.split("####")[-1].strip()
    #     print(f"Predicted answer: {seq_answer}, Reference answer: {reference_ans}")
    #     seq_answer = GSM8kTaskUtils.polish_answer(seq_answer)
    #     print(f"2 Predicted answer: {seq_answer}, Reference answer: {reference_ans}")
    #     return seq_answer == reference_ans

    @staticmethod
    def evaluate(sequence_or_text, reference_txt, sample, start_idx=0, tokenizer=None, polish_reference=True):
        # gold_answer = sample["answer"]
        if polish_reference:
            reference_ans = GSM8kTaskUtils.polish_ground_truth(reference_txt)
        else:
            reference_ans = reference_txt
        gold_answer = f"${reference_ans}$"

        sequence_txt = get_text(sequence_or_text, tokenizer, start_idx=start_idx, skip_special_tokens=True)

        # put the gold answer in latex format

        score, extracted_info = verification_metric([gold_answer], [sequence_txt])
        is_correct = score == 1.0


        print("-------------------")
        if extracted_info is not None:
            extracted_gold, extracted_pred = extracted_info
        else:
            extracted_gold, extracted_pred = None, None

        print(f"Gold Answer: {gold_answer}")
        print(f"Extracted Gold: {extracted_gold}")
        print(f"Extracted Pred: {extracted_pred}")
        print(f"Is Correct: {is_correct}")
        print("-------------------")
        return is_correct



    @staticmethod
    def prepare_sample(
        train_data, data_index, tokenizer, n_shot=1, enable_thinking=False
    ):
        qna = train_data[data_index]
        nshot_data = train_data[:data_index] + train_data[data_index + 1 :]
        prompt = nshot_chats(nshot_data=nshot_data, n=n_shot, question=qna["question"])
        ground_truth = qna["answer"]
        current_sequence = tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True, tokenize=True, return_tensors="pt",
            enable_thinking=enable_thinking
        )
        print(f"The prompt: {tokenizer.decode(current_sequence[0], skip_special_tokens=False)}")
        return current_sequence, ground_truth, prompt, qna

    @staticmethod
    def get_question(sample):
        return sample["question"]

        
    @staticmethod
    def polish_answer(answer):
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

    @staticmethod
    def check_soundness(sample, response_text):
        final_prompt = soundness_check_prompt_template.format(
            question=sample["question"],
            ground_truth=sample["answer"],
            response=response_text
        )

        return GSM8kTaskUtils.call_api(
            prompt=final_prompt
        )