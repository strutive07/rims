####! this must preserve the original order of the records as it needs to match with the dataset to score ####

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import fire
import jsonlines as jsl
import pandas as pd

# from run_baseline import save_res, dedup
from query_scenario import rims_query
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
    prompt_f: str = "",
    n: int = 1,
    temperature: float = 0.0,
    backbone: str = "vllm",
    seed: int = 777,
    # dataset_type: Literal["gsm", "ocw", "math"] = "",
):
    task_runner_obj = TaskRunner(80)

    for row in records:
        jobs = rims_query(
            row=row,
            prompt_f=prompt_f,
            n=n,
            temperature=temperature,
            backbone=backbone,
            seed=seed,
            # dataset_type=dataset_type
        )
        task_runner_obj.add_task(jobs)

    res = await task_runner_obj.run()
    return res


async def main(
    indiv_processed_jslf: Union[str, Path] = "",
    dataset_type: Optional[str] = "",
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

    if not dataset_type:
        print(f"{dataset_type=} not passed. Trying to infer from the file")
        dataset_type = list(jsonlines.open(indiv_processed_jslf))[0]["dataset_type"]

    prompt_files = Path("query_obj/rims_prompts/").glob(f"rims_{dataset_type}[0-9].txt")

    for prompt_f in prompt_files:
        print(f"Running for {prompt_f=}")
        with jsonlines.open(indiv_processed_jslf) as f:
            records = list(f)[start_idx:]
            records, removed_idxs = dedup(records)  # for sanity check.
            if removed_idxs:
                print(f"Removed {len(removed_idxs)} duplicates")
                print(removed_idxs)
            to_select_records, torun_idxs = filter_tobe_run(records)
        error_idx = []
        res = []

        outdir = Path(indiv_processed_jslf).parent / "rims" / Path(prompt_f).stem

        if not outdir.exists():
            outdir.mkdir(parents=True)

        outpath = outdir / f"n{n}_{temperature}_rims_raw_query_result.jsonl"

        res_selection = await run_task(
            prompt_f=prompt_f,
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

        # copy input file for postprocessing
        copypath = outdir / f"{outpath.stem}_input.jsonl"
        with jsl.open(copypath, "w") as writer:
            writer.write_all(records)

        # make copy of rims prompt used
        copypath_p = outdir / Path(prompt_f).name
        copypath_p.open("w").write(open(prompt_f).read())

        # to-postprocess-with are as follows:
        print(outpath)
        print(selection_idx_path)
        print(copypath)


if __name__ == "__main__":
    fire.Fire(main)
