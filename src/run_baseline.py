import asyncio
import json
import os
from pathlib import Path
from typing import Literal

import fire
import pandas as pd
from query_scenario import indiv_query
from task_runner import TaskRunner


def dedup(records):
    df = pd.DataFrame(records)
    k = "question" if "question" in df.columns else "problem"
    df = df.drop_duplicates(subset=k)
    records = df.to_dict(orient="records")
    return records


def save_res(outpath, res):
    error_rows = []
    with open(outpath, "w", encoding="utf-8") as f:
        for idx, row in enumerate(res):
            save_keys = ["contents", "query_message", "meta"]
            if "error" in row:
                error_rows.append(idx)
                save_obj = row
            else:
                save_obj = {
                    obj: {
                        save_key: row[obj].get(save_key, "") for save_key in save_keys
                    }
                    for obj in row.keys()
                }
            f.write(json.dumps(save_obj, ensure_ascii=False) + "\n")

    print("Error row count:", len(error_rows))
    print("Error rows:", error_rows)


async def run_task(
    records,
    n,
    temperature,
    p2c_plan_temperature,
    backbone,
    dataset_type,
    seed,
    error_idx,
):
    task_runner_obj = TaskRunner(100)

    for idx, record in enumerate(records):
        if len(error_idx) == 0 or idx in error_idx:
            jobs = indiv_query(
                record,
                n=n,
                temperature=temperature,
                p2c_plan_temperature=p2c_plan_temperature,
                seed=seed,
                backbone=backbone,
                dataset_type=dataset_type,
            )
            task_runner_obj.add_task(jobs)

    res = await task_runner_obj.run()
    return res


async def main(
    gsm_jslf: str = "",
    dataset_type: Literal[
        "gsm", "ocw", "math", "svamp"
    ] = "gsm",  # affects get_concordant_answer
    num_methods: int = 3,  # number of methods (3-> cot pal p2c / 2-> cot pal )
    start_idx: int = 0,
    # llm options
    n: int = 1,
    backbone: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    seed: int = 777,
    temperature: float = 0.0,
    p2c_plan_temperature: float = 0.0,
):
    assert gsm_jslf, f"need to specify {gsm_jslf=}"
    assert dataset_type in "gsm ocw math svamp".split(), f"invalid {dataset_type=}"

    if dataset_type == "svamp":
        with open(gsm_jslf) as f:
            records = json.load(f)
    else:
        import jsonlines

        with jsonlines.open(gsm_jslf) as f:
            records = list(f)[start_idx:]
            records = dedup(records)
    error_idx = []
    res = []

    outdir = Path("outputs") / f"{Path(gsm_jslf).stem}_dt.{dataset_type}" / backbone

    if not outdir.exists():
        outdir.mkdir(parents=True)

    outpath = outdir / f"n{n}_baseline_raw_query_result.jsonl"

    if outpath.exists():
        error_idx = []
        res = []
        import jsonlines
        with jsonlines.open(outpath) as f:
            for idx, row in enumerate(f):
                if "error" in row:
                    error_idx.append(idx)
                res.append(row)
        if len(error_idx) == 0:
            print('Tasks that have already been completed.')
            return
        else:
            print(f'Run {len(error_idx)} error_idx')

    res_current = await run_task(
        records,
        n,
        temperature,
        p2c_plan_temperature,
        backbone,
        dataset_type,
        seed,
        error_idx,
    )

    if error_idx:
        for idx, row in zip(error_idx, res_current):
            res[idx] = row
    else:
        res = res_current

    save_res(outpath, res)


if __name__ == "__main__":
    fire.Fire(main)
