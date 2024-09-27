import ast
import json
import os
import re
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from routellm.controller import ModelPair

"""
The core code is adapted to work with ARC dataset.
"""

INVALID = -9999999
VALID_LABELS = ["A", "B", "C", "D"]
ROUTED_PAIR = ModelPair(
    strong="gpt-4-1106-preview", weak="mistralai/Mixtral-8x7B-Instruct-v0.1"
)

class CustomBackendClient:
    def __init__(self, backend_name, base_url):
        self.backend_name = backend_name
        self.base_url = base_url

    def send_request(self, prompt):
        # Simulate or implement API call to custom backend here
        print(f"Sending request to {self.base_url} with prompt: {prompt}")
        # Implement actual HTTP request code here (e.g., using requests or httpx)
        return {"answer": "{\"answerKey\": \"A\"}"}  # Simulated response, replace with actual logic

    def get_chat_template(self):
    # 返回一个包含 default_system_prompt 的字典
        return {
            "model": self.backend_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": ""}
            ],
            "default_system_prompt": "Default system prompt here"  # 确保包含这个键
        }
    
    def end_program(self,*args):
        # Simulated cleanup or finalization process
        # print(f"Ending program for {self.backend_name}.")
        # Implement any necessary logic to clean up after a program
        pass

def select_sglang_backend(args):
    if args.backend.startswith("gpt") or args.backend.startswith("router-"):
        # Create a client to interact with the custom backend
        backend = CustomBackendClient(args.backend, base_url=f"{args.host}:{args.port}/v1")
    else:
        raise ValueError(f"Invalid backend: {args.backend}")
    return backend

def read_jsonl(filename: str):
    """Read a JSONL file."""
    rets = []
    with open(filename) as fin:
        for line in fin:
            if line.startswith("#"):
                continue
            rets.append(json.loads(line))
    return rets

def get_one_example(lines, i, include_answer):
    # Format for ARC multiple-choice questions
    question = lines[i]["question"]
    choices = lines[i]["choices"]["text"]
    choice_labels = lines[i]["choices"]["label"]
    
    # Format the question with the choices
    ret = f"Question: {question}\nChoices: {', '.join([f'{label}: {choice}' for label, choice in zip(choice_labels, choices)])}\nAnswer:"
    
    if include_answer:
        ret += f" {lines[i]['answerKey']}"
    return ret

def get_few_shot_examples(lines, k):
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret

def get_answer_value(answer_str):
    # Since ARC dataset provides answer keys as letters (e.g., "A", "B", etc.), we do not need number parsing
    return answer_str

def main(args):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    lines = read_jsonl(f"{current_dir}/ARC_validation_data.jsonl")
    train = read_jsonl(f"{current_dir}/ARC_train_data.jsonl")

    # Construct prompts
    k = 8  # Few-shot examples
    few_shot_examples = get_few_shot_examples(train, k)

    questions = []
    labels = []
    for i in range(len(lines)):
        answer_key = lines[i]["answerKey"]
        if answer_key not in VALID_LABELS:
            # Skip this line if the answer key is invalid
            print(f"Skipping line {i} with invalid answer key: {answer_key}")
            continue
        questions.append(get_one_example(lines, i, False))
        labels.append(answer_key)  # ARC uses labels like A, B, C, D

    arguments = [{"question": q} for q in questions]

    #####################################
    ######### SGL Program Begin #########
    #####################################
    import sglang as sgl

    @sgl.function
    def few_shot_arc(s, question):
        s += sgl.user(few_shot_examples + question)
        s += sgl.assistant(sgl.gen("answer", max_tokens=1024, stop=["Question"]))

    #####################################
    ########## SGL Program End ##########
    #####################################
    # Select backend
    backend = select_sglang_backend(args)

    # Run requests
    states = few_shot_arc.run_batch(
        arguments,
        temperature=0,
        backend=backend,
        num_threads=args.parallel,
        progress_bar=True,
    )

    preds = []
    responses = []
    for i, state in enumerate(states):
        # 直接访问 state 对象的 answer 属性
        answer = state.answer
        try:
            response = json.loads(answer)
            preds.append(response.get("answerKey", "D"))  # 默认值设为"D"
            responses.append(response.get("answerKey", "D"))
        except json.JSONDecodeError:
            print(f"Error decoding JSON response: {answer}")
            preds.append("D")  # 使用默认值
            responses.append("D")

    # Compute accuracy (labels and preds are single-character answers like A, B, C, D)
    accuracy = np.mean(np.array(preds) == np.array(labels))
    return accuracy, responses

evaluate_args_base = {
    "parallel": 64,
    "host": "http://localhost",
    "port": "6060",
}
weak_cors, weak_responses = main(
    SimpleNamespace(**evaluate_args_base, backend="router-random-1.0"),
)
strong_cors, strong_responses = main(
    SimpleNamespace(**evaluate_args_base, backend="router-random-0.0"),
)
current_dir = os.path.dirname(os.path.abspath(__file__))
prompts = pd.read_json(f"{current_dir}/ARC_validation_data.jsonl", lines=True)["question"].tolist()

assert len(weak_cors) == len(strong_cors)

result_df = pd.DataFrame(
    zip(prompts, weak_cors, strong_cors, weak_responses, strong_responses),
    columns=[
        "prompt",
        ROUTED_PAIR.weak,
        ROUTED_PAIR.strong,
        f"{ROUTED_PAIR.weak}_response",
        f"{ROUTED_PAIR.strong}_response",
    ],
)

current_dir = os.path.dirname(os.path.abspath(__file__))
result_df.to_csv(
    f"{current_dir}/arc_responses.csv",
    index=False,
)