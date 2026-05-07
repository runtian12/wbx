from abc import ABC, abstractmethod
import os
import json
import litellm
import sys
from typing import Dict, Any, List
import re


def extract_json_span(text: str) -> str:
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1 or end < start:
        raise ValueError("No '{...}' span found (first '{' comes after last '}' or braces missing).")
    return text[start:end+1]

def polish_json_response(json_str: str) -> str:
    """Clean JSON string by escaping unescaped backslashes to fix parsing errors."""
    # Use regex to find backslashes that are not already escaped
    # This matches a backslash that is not preceded by another backslash
    import re
    return re.sub(r'(?<!\\)\\(?!["\\\/bfnrt])', r'\\\\', json_str)

def get_json(text: str) -> Dict[str, Any]:
    """Extract and parse JSON object from a string."""
    json_span = extract_json_span(text)
    try:
        return json.loads(json_span)
    except:
        polished_json = polish_json_response(json_span)
        return json.loads(polished_json)


class BaseTaskUtils(ABC):
    """
    Base class for task-specific utility functions.
    """

    @staticmethod
    @abstractmethod
    def get_task_name():
        pass
    
    @staticmethod
    @abstractmethod
    def load_dataset(base_path):
        pass

    @staticmethod
    @abstractmethod
    def polish_ground_truth(txt):
        pass

    @staticmethod
    @abstractmethod
    def evaluate(sequence_or_text, reference_txt, sample, start_idx=0, tokenizer=None, polish_reference=True):
        pass

    @staticmethod
    @abstractmethod
    def prepare_sample(train_data, data_index, tokenizer):
        pass

    @staticmethod
    @abstractmethod
    def get_question(sample):
        pass
        
    @staticmethod
    @abstractmethod
    def polish_answer(answer, tokenizer=None, start_idx=0):
        pass

    @staticmethod
    @abstractmethod
    def check_soundness(sample, response_text):
        pass

    @staticmethod
    def call_api(prompt: str, reasoning_effort: str = "medium") -> Dict[str, Any]:
        """Sends the data to Gemini for analysis using litellm with retry logic."""

        # print(f"Final prompt for Gemini:\n{prompt}\n")
        messages = [{"content": prompt, "role": "user"}]
        print("*******************************")
        print(f"Sending request to Gemini with messages: {messages}")
        
        for attempt in range(1, 6):  # 5 attempts
            try:
                print(f"Sending request to Gemini (attempt {attempt}/5)...")
                response = litellm.completion(
                #     model="gemini/gemini-2.5-flash",
                    model="gemini/gemini-2.5-flash-lite",
                    messages=messages,
                    response_format={"type": "json_object"},
                    num_retries=1,
                    reasoning_effort=reasoning_effort,
                )
                print(f"response: {response}")
                
                # json_span = extract_json_span(response.choices[0].message.content)
                # analysis_json = json.loads(polished_json)
                # polished_json = polish_json_response(json_span)
                # analysis_json = json.loads(polished_json)
                # analysis_json = json.loads(response.choices[0].message.content)
                analysis_json = get_json(response.choices[0].message.content)
                print(f"Successfully parsed JSON on attempt {attempt}")
                return analysis_json
                
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"JSON parsing failed on attempt {attempt}: {e}")
            except Exception as e:
                print(f"API call failed on attempt {attempt}: {e}")
            
            if attempt < 5:
                print(f"Retrying... ({attempt + 1}/5)")
        
        # All attempts failed
        return {
            "analysis": "Failed to get valid response after 5 attempts",
            "decision": None
        }