# start from processed_indiv

import asyncio
from typing import Dict, List, Union

from query_obj import RimsQueryObject


async def rims_query(
    # indiv_records_path:Union[str, Path]="some.jsonl",
    row: Dict = None,
    prompt_f: str = "",
    temperature: float = 0.0,
    n: int = 1,
    seed: int = 777,
    backbone: str = "",
    stop: Union[str, List[str]] = None,
    disable_hinting: bool = False
) -> Dict:
    dataset_type = row["dataset_type"]
    assert dataset_type in "gsm ocw math svamp".split(), f"check {dataset_type=}"
    assert row["need_selection"][0], f"{row['need_selection']=} shouldn't reach here"
    assert backbone, f"{backbone=} must be passed!"

    # init
    rims_prompt_tmp = open(prompt_f).read().strip()

    if disable_hinting:
        rims_prompt_tmp = '\n'.join([
            row.replace(' and `Hint for a better Method choice`', '').replace(" and reason to take `Workaround Method` by writing `Hint for a better Method choice`", ' and choose `Workaround Method`')
            for row in rims_prompt_tmp.split('\n')
            if not row.startswith('`Hint for a better Method choice`:')
        ])
        
    query_obj = RimsQueryObject(
        # dataset_type=dataset_type,
        prompt_tmp=rims_prompt_tmp,
    )

    # placeholders
    return_data = {}
    jobs = []

    question = row["question"]

    max_tokens = 2048

    if stop is None:  # decode until it faces correct answer
        stop = [
            "\n`Evaluation`: Correct",
            "`Evaluation`: Correct",
            "Evaluation: Correct",
        ]

    query_params = {
        "question": question,
        "temperature": temperature,
        "backbone": backbone,
        "n": n,
        "seed": seed,
        "max_tokens": max_tokens,
        "stop": stop,
    }

    jobs.append(query_obj.async_query(**query_params))

    for contents, query_message, resp, meta in await asyncio.gather(*jobs):
        meta["gt_answer"] = row["gt_answer"]
        return_data[meta["method_obj"]] = {
            "contents": contents,  # already listified
            "query_message": query_message,
            "resp": resp,
            "meta": meta,
        }
    return return_data
