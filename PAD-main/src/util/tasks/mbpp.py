import re
from typing import Optional
from src.util.json_io import load_jsonlines
from src.util.tasks.base import BaseTaskUtils
from datasets import load_dataset, concatenate_datasets
import ast
from src.util.code_eval import compute_code_eval
from collections import Counter
from src.util.general import get_text

soundness_check_prompt_template = """
You are an expert code evaluator specializing in identifying flawed logic and poor practices in code solutions. Your task is to analyze a given code response and determine if it is sound, even if it passes some basic test cases.

Read the entire 'Response to Analyze' from start to finish. A response is considered to have **flawed reasoning** if you identify any of the following patterns:

- **Flawed Algorithm/Logic:** The code uses an incorrect algorithm or contains logical errors (e.g., wrong conditional, incorrect update logic) that may coincidentally produce a correct output for some test cases but is fundamentally unsound.
- **Failure to Handle Edge Cases:** The code works for typical inputs but would obviously fail on common edge cases (e.g., empty arrays, single-element arrays, zeros, or other special values) that a robust solution should handle.
- **Self-Correction:** The response contains code that is later deleted, commented out, or rewritten, often with comments like "that didn't work" or "a better way is...". This indicates a flawed initial approach.
- **Unnecessary or Redundant Code:** It includes variables, loops, conditions, or imported libraries that are declared but never used or do not contribute to the final solution, indicating a confused or inefficient thought process.

---

### **Examples**

**EXAMPLE 1: FLAWED REASONING (Flawed Algorithm)**

* **Question:** Write a function `min_jumps(arr)` to find the minimum number of jumps to reach the end of an array.
* **Golden Answer (for your reference):** A correct $O(n)$ greedy solution.
* **Response to Analyze:**
    ```python
    def min_jumps(arr):
        jumps = 0
        current_index = 0
        while current_index < len(arr) - 1:
            jumps += 1
            farthest_jump = current_index + arr[current_index]
            # This logic is flawed because it only considers the jump from the current spot.
            current_index = farthest_jump
        return jumps
    ```
* **Your JSON Output:**
    ```json
    {{
      "analysis": "The reasoning is flawed. The algorithm incorrectly assumes the best strategy is always to jump as far as possible from the current position. It fails to check other positions within the current jump's reach that might offer a better subsequent jump. This greedy approach is naive and incorrect.",
      "decision": true
    }}
    ```

**EXAMPLE 2: SOUND REASONING (No Flaws)**

* **Question:** Write a function `find_max(nums)` that returns the maximum element in a list of numbers.
* **Golden Answer (for your reference):** `return max(nums)`
* **Response to Analyze:**
    ```python
    def find_max(nums):
        if not nums:
            return None
        max_val = nums[0]
        for num in nums[1:]:
            if num > max_val:
                max_val = num
        return max_val
    ```
* **Your JSON Output:**
    ```json
    {{
      "analysis": "The reasoning is sound. The code correctly handles the edge case of an empty list and then iterates through the list, efficiently keeping track of the maximum value found so far. The logic is direct and robust.",
      "decision": false
    }}
    ```

**EXAMPLE 3: FLAWED REASONING (Unnecessary Code)**

* **Question:** Write a function `sum_positives(nums)` that sums all the positive numbers in a list.
* **Golden Answer (for your reference):** `return sum(n for n in nums if n > 0)`
* **Response to Analyze:**
    ```python
    import math # Unnecessary import

    def sum_positives(nums):
        total = 0
        neg_count = 0 # Unused variable
        for num in nums:
            if num > 0:
                total += num
            else:
                neg_count += 1
        return total
    ```
* **Your JSON Output:**
    ```json
    {{
      "analysis": "The reasoning is flawed because it contains unnecessary components that indicate a confused thought process. The 'math' library is imported but never used, and the 'neg_count' variable is calculated but serves no purpose for the final result.",
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

After carefully reviewing the 'Response to Analyze', provide your evaluation in a strict JSON format, following the structure shown in the examples above. The JSON object must contain exactly two keys: `"analysis"` and `"decision"`. The `"decision"` should be `true` if flawed reasoning is present and `false` otherwise.

**Your JSON Output:**
"""



def find_duplicate_functions(code_string):
    """
    Finds function names defined more than once in a string containing Python code.

    Args:
        code_string: A string containing Python code snippets.

    Returns:
        A list of function names that are defined multiple times.
        Returns an empty list if no duplicates are found.
    """
    # Regex to find lines starting with 'def ' (potentially indented)
    # and capture the function name.
    # It looks for 'def', followed by whitespace, then captures the function name
    # (letters, numbers, underscore, starting with letter/underscore),
    # followed by optional whitespace and an opening parenthesis '('.
    # re.MULTILINE makes '^' match the start of each line.
    function_names = re.findall(r"^\s*def\s+([a-zA-Z_]\w*)\s*\(", code_string, re.MULTILINE)

    if not function_names:
        return [] # No function definitions found

    # Count the occurrences of each function name
    name_counts = Counter(function_names)

    # Filter for names that appear more than once
    duplicate_functions = [name for name, count in name_counts.items() if count > 1]

    return duplicate_functions

def generate_prompt_mbpp(sample):
    tree = ast.parse(sample["code"])
    func_def = None
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef):
            if (
                node.name
                == sample["test_list"][0].split("assert")[1].split("(")[0].strip()
            ):
                func_def = ast.unparse(node).split("\n")[0]
                break
    if func_def is None:
        print("Cannot extract function definition. Skipping sample...")
        return None
    test_cases = "\n".join(sample["test_list"])
    # test_cases = sample["test_list"][0]
    return f'Function definition: {sample["text"]}\nFunction signature: {func_def}\nTest case: {test_cases}\nPython code:'


def get_function_signature(starter_code):
    lines = starter_code.splitlines()
    for line in lines:
        if line.strip().startswith("def"):
            return line.replace("def", "").replace(":", "").split("->")[0].strip()
    return None


def parse_code_block(string: str, lang: str) -> Optional[str]:
    code_pattern = rf"```{lang}\n(.*?)\n```"
    match = re.search(code_pattern, string, re.DOTALL)

    if match:
        return match.group(1)

    generic_code_pattern = r"```\n(.*?)\n```"
    match = re.search(generic_code_pattern, string, re.DOTALL)

    if match:
        return match.group(1)

    return parse_first_func(string, lang)


def parse_first_func(code: str, lang: str) -> Optional[str]:
    assert lang == "python", "Only python is supported for now. TODO: Rust"
    code_lines = code.split("\n")
    def_i = -1
    last_i = 0
    got_return = False
    for i, line in enumerate(code_lines):
        if line.startswith("def "):
            if def_i == -1:
                def_i = i
            else:
                break
        elif "return" in line and def_i != -1:
            got_return = True
        if line == "" and def_i != -1 and got_return:
            last_i = i
            break

    if last_i == 0:
        last_i = len(code_lines) - 1

    if def_i == -1:
        return None

    return "\n".join(code_lines[def_i : last_i + 1]).rstrip("[/PYTHON]")



class MBPPTaskUtils(BaseTaskUtils):
    @staticmethod
    def get_task_name():
        return "mbpp"

    @staticmethod
    def load_dataset(base_path=None):
        dataset = load_dataset("google-research-datasets/mbpp")
        temp_test = dataset["test"].shuffle(seed=42)
        test_data = temp_test.select(range(200))
        remaining_test_for_train = temp_test.select(range(200, len(temp_test)))
        train_data = concatenate_datasets(
            [
                dataset["train"],
                dataset["validation"],
                # dataset["prompt"],
                remaining_test_for_train,
            ]
        )
        return train_data, test_data

    # @staticmethod
    # def evaluate(sequence_or_text, reference_txt, sample, start_idx=0, tokenizer=None, polish_reference=True):
    #     if polish_reference:
    #         reference_ans = GSM8kTaskUtils.polish_ground_truth(reference_txt)
    #     else:
    #         reference_ans = reference_txt

    #     sequence_txt = get_text(sequence_or_text, tokenizer, start_idx=start_idx, skip_special_tokens=True)
    #     seq_answer = sequence_txt.split("####")[-1].strip()
    #     seq_answer = GSM8kTaskUtils.polish_answer(seq_answer)
    #     return seq_answer == reference_ans

    @staticmethod
    def polish_ground_truth(txt):
        return txt

    @staticmethod
    def evaluate(sequence_or_text, reference_txt, sample, start_idx=0, tokenizer=None, polish_reference=True):
        sequence_txt = get_text(sequence_or_text, tokenizer, start_idx=start_idx, skip_special_tokens=True)
        eval_res = compute_code_eval(
            references=["\n".join(sample["test_list"])],
            predictions=[[parse_code_block(sequence_txt, "python")]],
            k=1,
        )
        correctness = eval_res[0]["pass@1"]
        correctness = float(correctness)

        return correctness == 1.0

    @staticmethod
    def prepare_sample(train_data, data_index, tokenizer, enable_thinking=False):
        system_prompt = "You are an AI that only responds with python code, NOT ENGLISH. You will be given a function definition, its signature and a simple test case by the user. Write your full implementation (restate the function signature). Only output the code, do not add explanations or comments in the code. Do not include any test cases in your response.\n\n"
        code_block = "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"

        sample = train_data[data_index]
        prompt = generate_prompt_mbpp(sample)
        ground_truth = sample["code"]
        messages = [
            {"role": "system", "content": f"{system_prompt}\n{code_block}"},
            {"role": "user", "content": prompt},
        ]
        current_sequence = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt",
            enable_thinking=enable_thinking
        )
        return current_sequence, ground_truth, messages, sample
    
    @staticmethod
    def get_question(sample):
        return generate_prompt_mbpp(sample)

    @staticmethod
    def get_ground_truth(sample):
        return sample["code"]

    @staticmethod
    def polish_answer(answer):
        return answer
    
    @staticmethod
    def check_soundness(sample, response_text):
        duplicated_functions_sequence = find_duplicate_functions(response_text)
        if len(duplicated_functions_sequence) > 0:
            return {
                "analysis": "Duplicated function definitions found.",
                "decision": True,  # if a flaw is found
            }

        question = MBPPTaskUtils.get_question(sample)
        ground_truth = MBPPTaskUtils.get_ground_truth(sample)
        final_prompt = soundness_check_prompt_template.format(
            question=question,
            ground_truth=ground_truth,
            response=response_text
        )
        return MBPPTaskUtils.call_api(final_prompt, reasoning_effort="low")
