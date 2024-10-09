import re
from pathlib import Path
from typing import Dict, List, Literal

import jsonlines as jsl
from fire import Fire
from processings.text_exec_functions import (
    extract_ans_from_cot_MATHnOCW,
    extract_num_turbo,
    get_concordant_answer,
    get_concordant_answer_n,
    safe_execute_turbo,
)
from processings.text_parse_functions import postprocess_code
from tqdm import tqdm


def process_indiv(
    infile: str = "raw_indiv.jsonl",
    outfile: str = "processed_indiv.jsonl",
):
    """
    processed_indiv.jsonl
    row:
    {
    "question":str,
    "answer":str,

    "cot_pred": List,
    "pal_pred": List,
    "p2c_pred": List,
    "majvote_answers": List,

    "cot_solution": List[str],
    "pal_solution": List[str],
    "p2c_solution": List[str],
    }

    """
    parent_dir = Path(infile).parent
    print(f"parent dir is automatically set to {parent_dir=}")
    print(f"outfile is set to {Path(outfile).name} and saved under {parent_dir=}")
    infile = Path(infile).name
    outfile = Path(outfile).name
    outjslf = Path(parent_dir) / outfile
    assert infile != outfile

    if outjslf.exists():
        print(outjslf, ': Tasks that have already been completed.')
        return

    records = list(jsl.open(parent_dir / infile))

    dataset_type = records[0]["CoTQueryObject"]["meta"]["dataset_type"]
    processed_rows = []
    for row in tqdm(records):
        question = row["CoTQueryObject"]["query_message"][-1]["content"].replace(
            "Question: ", ""
        )
        # answer = row["answer"]

        # regardless of n,
        raw_cots = row["CoTQueryObject"]["contents"]
        raw_pals = row["PALQueryObject"]["contents"]
        raw_p2cs = row["P2CQueryObject"]["contents"]

        # solutions
        cot_solutions = raw_cots
        pal_solutions = [postprocess_code(r) for r in raw_pals]
        p2c_solutions = [postprocess_code(r) for r in raw_p2cs]

        # executions
        cot_exec = (
            extract_num_turbo
            if dataset_type == "gsm"
            else extract_ans_from_cot_MATHnOCW
        )
        
        code_exec = safe_execute_turbo
    
        cot_preds = [cot_exec(s) for s in cot_solutions]
        pal_preds = [code_exec(s) for s in pal_solutions]
        p2c_preds = [code_exec(s) for s in p2c_solutions]
        

        majvote_answers = [
            get_concordant_answer([c, p, p2], dataset_type=dataset_type)
            for c, p, p2 in zip(cot_preds, pal_preds, p2c_preds)
        ]  # List[Union[str,float,None]]

        # to run selection
        need_selection = [maj is None for maj in majvote_answers]

        # for later ease of scoring
        gt_answer = row["CoTQueryObject"]["meta"]["gt_answer"]

        processed_row = dict(
            question=question,
            # answer = answer,
            cot_solutions=cot_solutions,
            pal_solutions=pal_solutions,
            p2c_solutions=p2c_solutions,
            cot_preds=cot_preds,
            pal_preds=pal_preds,
            p2c_preds=p2c_preds,
            majvote_answers=majvote_answers,
            need_selection=need_selection,
            dataset_type=dataset_type,
            gt_answer=gt_answer,
        )
        processed_rows.append(processed_row)

    if not outjslf.parent.is_dir():
        outjslf.parent.mkdir(parents=True, exist_ok=True)

    with jsl.open(outjslf, "w") as writer:
        writer.write_all(processed_rows)
        print(f"wrote {len(processed_rows)} rows to")
        print("\t", outjslf)


def process_cross_and_mix(
    dataset_type: str = "",
    infile: str = "",
    outfile: str = "processed_cross_and_mix.jsonl",
    n: int = 1,
):
    assert dataset_type != ""
    assert infile, f"need to specify {infile=}"

    # path
    parent_dir = Path(infile).parent
    print(f"parent dir is automatically set to {parent_dir=}")
    print(f"outfile is set to {Path(outfile).name} and saved under {parent_dir=}")
    infile = Path(infile).name
    outfile = Path(outfile).name
    outjslf = Path(parent_dir) / outfile
    assert infile != outfile
    # if outjslf.exists():
    #     print(outjslf, ': Tasks that have already been completed.')
    #     return

    if n > 1 or not infile.startswith("n1_"):
        raise NotImplementedError("n>1 cannot run here")

    raw_selections = list(jsl.open(parent_dir / infile))
    selidxs = [
        int(k)
        for k in open(parent_dir / "selection_performed_idxs.txt")
        .read()
        .strip()
        .split("\n")
    ]

    assert len(raw_selections) == len(selidxs)
    originals = list(jsl.open(parent_dir / infile.replace(".jsonl", "_input.jsonl")))

    def _process_cross_and_mix_raw(sel: dict = None, og: dict = None) -> dict:
        cross_and_mix_res = sel['CrossAndMixQueryObject']['contents'][0]

        if cross_and_mix_res is None:
            og["cross_and_mix_answer"] = [None]
        else:
            cot_exec = (
                extract_num_turbo
                if dataset_type == "gsm"
                else extract_ans_from_cot_MATHnOCW
            )

            og["cross_and_mix_answer"] = cot_exec(cross_and_mix_res)
        return og

    for idx, selrow in zip(selidxs, raw_selections):
        originals[idx] = _process_cross_and_mix_raw(sel=selrow, og=originals[idx])

    # save
    if not outjslf.parent.is_dir():
        outjslf.parent.mkdir(parents=True, exist_ok=True)

    with jsl.open(outjslf, "w") as writer:
        writer.write_all(originals)
        print("\t", outjslf)

def process_simple_greedy(
    infile: str = "",
    outfile: str = "processed_sg.jsonl",
    n: int = 1,
):
    assert infile, f"need to specify {infile=}"

    # path
    parent_dir = Path(infile).parent
    print(f"parent dir is automatically set to {parent_dir=}")
    print(f"outfile is set to {Path(outfile).name} and saved under {parent_dir=}")
    infile = Path(infile).name
    outfile = Path(outfile).name
    outjslf = Path(parent_dir) / outfile
    assert infile != outfile
    if outjslf.exists():
        print(outjslf, ': Tasks that have already been completed.')
        return
        

    if n > 1 or not infile.startswith("n1_"):
        raise NotImplementedError("n>1 cannot run here")

    raw_selections = list(jsl.open(parent_dir / infile))
    selidxs = [
        int(k)
        for k in open(parent_dir / "selection_performed_idxs.txt")
        .read()
        .strip()
        .split("\n")
    ]

    assert len(raw_selections) == len(selidxs)
    originals = list(jsl.open(parent_dir / infile.replace(".jsonl", "_input.jsonl")))

    # process
    import re

    def _find_first_abc(text):
        match = re.search(r"\((A|B|C)\)", text)
        return match.group() if match else None

    def _process_sg_raw(sel: dict = None, og: dict = None) -> dict:
        # read raw sel row and fill the og row with it.
        abc = _find_first_abc(sel["SimpleGreedyQueryObject"]["contents"][0])
        if abc is None:
            og["sg_answer"] = [None]  # failed to generate the selection
            og["sg_selected"] = "failed"
        elif abc == "(A)":
            og["sg_answer"] = og["cot_preds"]  # list
            og["sg_selected"] = "cot"  # list
        elif abc == "(B)":
            og["sg_answer"] = og["pal_preds"]
            og["sg_selected"] = "pal"
        elif abc == "(C)":
            og["sg_answer"] = og["p2c_preds"]
            og["sg_selected"] = "p2c"
        else:
            og["sg_answer"] = [None]  # failed to generate the selection
            og["sg_selected"] = "failed"

        return og

    for idx, selrow in zip(selidxs, raw_selections):
        originals[idx] = _process_sg_raw(sel=selrow, og=originals[idx])

    # save
    if not outjslf.parent.is_dir():
        outjslf.parent.mkdir(parents=True, exist_ok=True)

    with jsl.open(outjslf, "w") as writer:
        writer.write_all(originals)
        print("\t", outjslf)


def process_rims(
    ptn: str = "pattern or path/to/rawresult jsonl parent",
    # outfile: str = "processed_rims.jsonl",
    n: int = 1,
):
    print(n, type(n))
    assert ptn, f"need to specify {ptn=}"
    if n > 1:
        raise NotImplementedError(
            "n>1 will need some tweak (majority voting at the last...)"
        )

    parent_dirs = list(Path("./").glob(ptn))
    print(parent_dirs)

    # path
    for parent_dir in parent_dirs:
        infiles = list(parent_dir.glob("n[0-9]_*_rims_raw_query*_result.jsonl"))
        
        for infile in infiles:
            ogpath = parent_dir / infile.name.replace(".jsonl", "_input.jsonl")
            if 'disable_hinting' in infile.name:
                outfile = parent_dir / "processed_disable_hinting_rims.jsonl"
            else:
                outfile = parent_dir / "processed_rims.jsonl"
            if outfile.exists():
                print(outfile, ': Tasks that have already been completed.')
                continue
            
            assert infile != outfile
    
            raw_selections = list(jsl.open(infile))
            selidxs = [
                int(k)
                for k in open(parent_dir / "selection_performed_idxs.txt")
                .read()
                .strip()
                .split("\n")
            ]
    
            assert len(raw_selections) == len(selidxs)
            originals = list(jsl.open(ogpath))
            dataset_type = originals[0]["dataset_type"]
    
            # process
            from processings.text_exec_functions import process_rims_out_dict
            from processings.text_parse_functions import parse_raw_modif
    
            def _process_rims_raw(sel_content: str = None, og: dict = None) -> dict:
                # read raw sel row and fill the og row with it.
                og["rims_selected"] = None
                og["rims_solution"] = None
                og["rims_answer"] = None
                og["rims_summary"] = None
                # for error logging
                og["error"] = False
                og["raw_text"] = None
                og["exception"] = None
                try:
                    parsed = parse_raw_modif(sel_content)
                except Exception as e:  # parsing fails
                    og["error"] = True
                    og["exception"] = f"parse_raw_modif()  {str(e)}"
                    og["raw_text"] = sel_content
                else:  # parsing success
                    try:
                        executed = process_rims_out_dict(parsed)
                        og["rims_selected"] = executed["good_method"]
                        og["rims_solution"] = executed["good_solution"]
                    except Exception as e:  # parse -> code fail
                        print(e)
                        og["error"] = True
                        og["exception"] = f"process_rims_out_dict()  {str(e)}"
                        og["raw_text"] = sel_content
                    else:  # code success (all success)
                        og["rims_answer"] = executed["good_ans"]
                        og["rims_summary"] = executed
    
                return og
    
            err_idxs = []
            for idx, selrow in tqdm(zip(selidxs, raw_selections), total=len(selidxs)):
                originals[idx] = _process_rims_raw(
                    sel_content=selrow["RimsQueryObject"]["contents"][0], og=originals[idx]
                )
                if originals[idx]["error"]:
                    err_idxs.append(idx)
    
            # save
            outjslf = outfile
            errjslf = outfile.with_name(outfile.name.replace(".jsonl", "_errors.jsonl"))
    
            print(outjslf)
            print(errjslf)
            with jsl.open(outjslf, "w") as writer, jsl.open(errjslf, "w") as writer_err:
                writer.write_all(originals)
                writer_err.write_all([originals[i] for i in err_idxs])
                print(f"processed\n\t{infile}\n\t{ogpath}")
                print(f"to\n\t{outfile}")
                print(f"\t{errjslf}", f"{len(err_idxs)}/{len(originals)}", "errors")


if __name__ == "__main__":
    Fire()
