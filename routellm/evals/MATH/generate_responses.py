import ast
import json
import os
import re
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from openai import OpenAI

from routellm.controller import ModelPair

"""
The core code is based heavily on the original SGLang implementation.
"""

INVALID = -9999999
ROUTED_PAIR = ModelPair(
    strong="gpt-4-1106-preview", weak="mistralai/Mixtral-8x7B-Instruct-v0.1"
)


def select_sglang_backend(args):
    if args.backend.startswith("gpt") or args.backend.startswith("router-"):
        backend = OpenAI(args.backend, base_url=f"{args.host}:{args.port}/v1")
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
    """Format a single example."""
    problem = lines[i]["problem"]
    prompt = f"Problem: {problem}\nAnswer:"
    if include_answer:
        solution = lines[i]["solution"]
        answer = extract_answer_from_solution(solution)
        prompt += f" {answer}\n"
    return prompt


def extract_answer_from_solution(solution_str):
    """Extracts the final boxed answer from the solution string."""
    # This regex looks for the boxed answer pattern: \boxed{}
    match = re.search(r"\\boxed\{(.*?)\}", solution_str)
    if match:
        return match.group(1).strip()
    return INVALID


def get_few_shot_examples(lines, k):
    """Get few-shot examples from the training data."""
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret


def get_answer_value(answer_str):
    """Ensures the extracted answer is valid."""
    # In this case, we return the numeric or algebraic answer extracted from \boxed{}
    return answer_str.strip() if answer_str != INVALID else INVALID


def main(args):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    lines = read_jsonl(f"{current_dir}/test_math.jsonl")
    train = read_jsonl(f"{current_dir}/train_math.jsonl")

    # Construct prompts
    k = 8  # Number of few-shot examples
    few_shot_examples = get_few_shot_examples(train, k)

    questions = []
    labels = []
    for i in range(len(lines)):
        questions.append(get_one_example(lines, i, False))
        labels.append(extract_answer_from_solution(lines[i]["solution"]))
    assert all(l != INVALID for l in labels)
    arguments = [{"question": q} for q in questions]

    #####################################
    ######### SGL Program Begin #########
    #####################################

    import sglang as sgl

    @sgl.function
    def few_shot_math(s, question):
        s += sgl.user(few_shot_examples + question)
        s += sgl.assistant(sgl.gen("answer", max_tokens=1024, stop=["Problem"]))

    #####################################
    ########## SGL Program End ##########
    #####################################

    # Select backend
    backend = select_sglang_backend(args)

    # Run requests
    tic = time.time()
    states = few_shot_math.run_batch(
        arguments,
        temperature=0,
        backend=backend,
        num_threads=args.parallel,
        progress_bar=True,
    )

    preds = []
    responses = []
    for i in range(len(states)):
        preds.append(get_answer_value(states[i]["answer"]))
        responses.append(states[i]["answer"])

    # Compute accuracy
    return np.array(preds) == np.array(labels), responses


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
prompts = pd.read_json(f"{current_dir}/test_math.jsonl", lines=True)["problem"].tolist()

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
    f"{current_dir}/math_responses.csv",
    index=False,
)
