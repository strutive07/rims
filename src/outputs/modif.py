from pathlib import Path
from typing import Dict

import jsonlines as jsl
import pandas as pd

dataset_type_2_dspath = {
    "gsm": "/Users/seonils/dev/rims_minimal/dataset/gsm8K_test.jsonl",
    "math": "/Users/seonils/dev/rims_minimal/dataset/MATH/MATH-full.jsonl",
    # "ocw": "/Users/seonils/dev/rims_minimal/dataset/ocw/ocw_course.jsonl",
}


def insert_datatype_meta(records: list, dataset_type: str):
    # err_idx = []
    for i, row in enumerate(records):
        for qobj in row.keys():
            row[qobj]["meta"]["dataset_type"] = dataset_type
            # try:
            #     row[qobj]["meta"]["dataset_type"] = dataset_type
            # except Exception as e:
            #     print(e)
            #     err_idx.append(i)

    return records  # , err_idx


def insert_gt_answer_to_raw(
    records_raw: list, dataset_records: list, err_idx: list = None
) -> list:
    # first align two records according to its question
    def question_from_indiv_row(row: Dict = None) -> str:
        return (
            row["CoTQueryObject"]["query_message"][-1]["content"]
            .replace("Question: ", "")
            .strip()
        )

    # records_raw = [r for i, r in records_raw if r not in err_idx]

    df_raw = pd.DataFrame(records_raw)
    df_ds = pd.DataFrame(dataset_records).rename(
        columns=lambda x: "question" if x == "problem" else x
    )
    df_raw["question"] = df_raw.apply(question_from_indiv_row, axis=1)
    print((df_raw.question != df_ds.question).sum())
    print(df_raw[df_raw.question != df_ds.question].question)
    print(df_ds[df_raw.question != df_ds.question].question)
    assert (df_raw.question == df_ds.question).all()

    answers = df_ds.answer.tolist()
    for row_raw, answer in zip(records_raw, answers):
        for qobj in row_raw.keys():
            row_raw[qobj]["meta"]["gt_answer"] = answer
    return records_raw


# jsl of interests
for par in Path().glob("*"):
    dataset_type = par.suffix[1:]
    if dataset_type not in dataset_type_2_dspath:
        continue
    dataset_jsl = dataset_type_2_dspath[dataset_type]
    dataset_records = list(jsl.open(dataset_jsl))
    jslfs = list(par.glob("Phi-3*/n1*.jsonl"))
    # jslfs = list(par.glob("Meta-Llama*/n1*.jsonl"))
    for f in jslfs:
        print(f.name)
        records = list(jsl.open(f))
        # records, err_idx = insert_datatype_meta(records, dataset_type)
        records = insert_datatype_meta(records, dataset_type)
        records = insert_gt_answer_to_raw(records, dataset_records)
        with jsl.open(f"{str(f)}_", "w") as writer:
            writer.write_all(records)
