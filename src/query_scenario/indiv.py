"""
running
cot
pal
p2c

for a designated dataset



resources:

src/run_inference.py:indiv_inference()
src/utils/llm_query_utils.py
    src/utils/math_prompt.py # gsm cot/pal prompts
    src/utils/plancode_util_v2.py # plan2code prompts

"""
import asyncio
from typing import Literal

from query_obj import CoTQueryObject, P2CQueryObject, PALQueryObject


async def indiv_query(
    row: dict = None,
    num_methods: int = 3,
    temperature: float = 0.0,
    p2c_plan_temperature: float = 0.0,
    n: int = 1,
    seed: int = 777,
    backbone: str = "chatgpt0613long",
    dataset_type: Literal["gsm", "ocw", "math", "svamp"] = "",
    only_retrieve: bool = False,  # if true, when undone thing found, throws error
):
    """
    inference each method and return indiv results
    if there are already existing results, use them.


    return:
        solmap : {cot: pal: p2c:}
        ansmap : {cot: pal: p2c:} (solution executed)
    """

    if dataset_type == "ocw":
        question = row["problem"]
    elif dataset_type == "svamp":
        question = row["Body"] + '\n' + row["Question"]
    else:
        question = row["question"]

    missing_methods = ["cot", "pal", "p2c"]

    if only_retrieve:
        raise ValueError(
            f"no existing results found while {only_retrieve=}\n\n{missing_methods=}\n{ansmap=}"
        )

    query_objects = {
        "cot": CoTQueryObject,
        "pal": PALQueryObject,
        "p2c": P2CQueryObject,
    }

    max_tokens = {
        "cot": {
            "gsm": 2048,
            "ocw": 2048,
            "math": 2048,
            "svamp": 2048,
        },
        "pal": {
            "gsm": 2048,
            "ocw": 2048,
            "math": 2048,
            "svamp": 2048,
        },
        "p2c": {
            "gsm": 2048,
            "ocw": 2048,
            "math": 2048,
            "svamp": 2048,
        },
    }

    return_data = {}

    jobs = []

    for method in missing_methods:
        # function prepare variables
        init_param = {"dataset_type": dataset_type}
        if method == "p2c":
            init_param["plan_temperature"] = p2c_plan_temperature

        query_params = {
            "question": question,
            "temperature": temperature,
            "backbone": backbone,
            "n": n,
            "seed": seed,
            "max_tokens": max_tokens[method][dataset_type],
            "stop": "\n\n\n",
        }

        if method == "p2c":
            query_params["stop"] = "Question: "

        query_obj = query_objects[method](**init_param)
        jobs.append(query_obj.async_query(**query_params))

    for contents, query_message, resp, meta in await asyncio.gather(*jobs):
        if meta["dataset_type"] == "svamp":
            meta["gt_answer"] = row["Answer"]
        else:
            meta["gt_answer"] = row["answer"]  # append it for later ease
        return_data[meta["method_obj"]] = {
            "contents": contents,
            "query_message": query_message,
            "resp": resp,
            "meta": meta,
        }
    return return_data
