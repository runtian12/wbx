import json

def load_jsonlines(file_name: str):
    with open(file_name, 'r') as f:
        return [json.loads(line) for line in f]

def save_line_jsonlines(file_name: str, data: dict):
    with open(file_name, 'a') as f:
        f.write(json.dumps(data) + '\n')