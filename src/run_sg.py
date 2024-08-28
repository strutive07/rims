####! this must preserve the original order of the records as it needs to match with the dataset to score ####

import json
from pathlib import Path
from typing import Dict, List, Union

import fire
import jsonlines as jsl
import pandas as pd

# from run_baseline import save_res, dedup
from query_scenario import simple_greedy_query
from task_runner import TaskRunner


def filter_tobe_run(records):
    df = pd.DataFrame(records).reset_index(drop=True)
    mask = df.need_selection.apply(lambda x: x[0])
    df = df.reset_index()
    records = df[mask].to_dict(orient="records")
    torun_idx = df[mask].index.tolist()
    return records, torun_idx


def dedup(records):
    df = pd.DataFrame(records)
    k = "question" if "question" in df.columns else "problem"
    df_ = df.drop_duplicates(subset=k)
    records = df_.to_dict(orient="records")
    removed_idxs = list(set(df.index) - set(df_.index))
    return records, removed_idxs


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

    print(outpath)
    print("Error row count:", len(error_rows))
    print("Error rows:", error_rows)


async def run_task(
    records: List[Dict] = None,
    n: int = 1,
    temperature: float = 0.0,
    backbone: str = "vllm",
    seed: int = 777,
    # dataset_type: Literal["gsm", "ocw", "math"] = "",
):
    task_runner_obj = TaskRunner(100)

    for row in records:
        jobs = simple_greedy_query(
            row=row,
            temperature=temperature,
            n=n,
            seed=seed,
            backbone=backbone,
            # dataset_type=dataset_type
        )
        task_runner_obj.add_task(jobs)

    res = await task_runner_obj.run()
    return res


async def main(
    indiv_processed_jslf: Union[str, Path] = "",
    start_idx: int = 0,
    # llm options
    n: int = 1,
    backbone: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    seed: int = 777,
    temperature: float = 0.0,
    # err_idx: list = None,
):
    """
    dataset_type will be included in every row of
        `indiv_processed_jslf`
    """
    assert indiv_processed_jslf, f"need to specify {indiv_processed_jslf=}"

    import jsonlines

    with jsonlines.open(indiv_processed_jslf) as f:
        records = list(f)[start_idx:]
        records, removed_idxs = dedup(records)  # for sanity check.
        if removed_idxs:
            print(f"Removed {len(removed_idxs)} duplicates")
            print(removed_idxs)
        to_select_records, torun_idxs = filter_tobe_run(records)
    error_idx = []
    res = []

    outdir = Path(indiv_processed_jslf).parent / "simple_greedy"

    if not outdir.exists():
        outdir.mkdir(parents=True)

    outpath = outdir / f"n{n}_{temperature}_sg_raw_query_result.jsonl"
    if outpath.exists():
        print('Tasks that have already been completed.')
        return

    res_selection = await run_task(
        records=to_select_records,
        n=n,
        temperature=temperature,
        backbone=backbone,
        seed=seed,
    )
    # save_results
    save_res(outpath, res_selection)

    # record idxs selection performed on
    selection_idx_path = outdir / "selection_performed_idxs.txt"
    with open(selection_idx_path, "w") as f:
        f.write("\n".join(map(str, torun_idxs)))

    # make symlink to input file for postprocessing
    copypath = outdir / f"{outpath.stem}_input.jsonl"
    with jsl.open(copypath, "w") as writer:
        writer.write_all(records)

    # to-postprocess-with are as follows:
    print(outpath)
    print(selection_idx_path)
    print(copypath)


if __name__ == "__main__":
    fire.Fire(main)

    # pd.concat([d1,d2], axis="index").sort_values(by="index").drop(columns="index").reset_index(drop=True)


"""
running
rims (ours) selection

** note: selection algorithm supposed to run only where majority vote failed to reach the consensus **


for a designated dataset

----
resources:
    src/run_inference.py

    prompts for
    gsm, ocw rims :

        GSM_RIMS_RW=prompt_construction_src/newer_prompts_3/renewed_gsm_prompts/rewrote.p2c_gsm_newer_best_p2c2cot.pal2p2c.pal2cot.txt
        GSM_RIMS_RW_1=prompt_construction_src/newer_prompts_3/renewed_gsm_prompts/rewrote.p2c_gsm_cot2p2c.pal2cot.pal2p2c.txt
        GSM_RIMS_RW_2=prompt_construction_src/newer_prompts_3/renewed_gsm_prompts/rewrote.p2c_gsm_pal2p2c.cot2p2c.cot2pal.txt

        OCW_RIMS=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_ocw_p2c-cot.pal-p2c.pal-cot__.txt
        OCW_RIMS_1=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_ocw_p2c-cot.pal-p2c.cot-p2c__.txt

    math rims: https://github.com/fgenie/rims_minimal/tree/m1mac/src/prompt_construction_src/ocw_math_prep_rims_prompt/prompts/newprompts
"""
