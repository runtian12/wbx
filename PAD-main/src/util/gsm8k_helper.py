import random

def nshot_chats(nshot_data: list, n: int, question: str, seed: int = None):

    def question_prompt(s):
        return f'Question: {s}'

    def answer_prompt(s):
        # Extract the numeric answer from the format "... #### 624"
        if '####' in s:
            calculation_steps = s.split('####')[0].strip()
            final_answer = s.split('####')[1].strip()
            return f'Answer: {calculation_steps}\nTherefore, the answer is \\boxed{{{final_answer}}}.'
        else:
            return f'Answer: {s}'

    chats = [
        # {
        #     "role": "system",
        #     "content": "You are a grade school math problem solver. At the end, you MUST write the answer as an integer after '####'. Let's think step by step.",
        # },
    ]
    # if seed is not None:
    #     random.seed(seed)
    # random.seed(42)
    for qna in random.sample(nshot_data, n):
        chats.append(
            {"role": "user", "content": question_prompt(qna["question"])})
        chats.append(
            {"role": "assistant", "content": answer_prompt(qna["answer"])})

    chats.append({"role": "user", "content": question_prompt(question)+" Please solve this step by step and provide your final answer as an integer in \\boxed{} format.\n\nStep-by-Step Answer:"})

    return chats