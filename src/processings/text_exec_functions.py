import queue  # designed thread-safe...
import re
from collections import Counter
from itertools import combinations
from typing import Dict, List, Literal, Union

err_queue = queue.Queue()  # manages errors in subthread

import func_timeout

# to avoid matplotlib error by display in code execution
import matplotlib
from processings import math_util

matplotlib.use("Agg")


######################
## code execution ##
######################


def _convert_to_float_if_possible(ans):
    try:
        return float(ans)
    except Exception:
        return ans


def _convert_to_str_if_not_none_nor_float(ans):
    if ans is not None and not isinstance(ans, float):
        try:
            return str(ans)
        except Exception as e:
            return ans
    else:
        return ans


def get_func_name_from_string(codestring: str) -> str:
    match = re.search(
        r"def (\w+)\(\):", codestring
    )  # kwargs만으로 정의되며 default가 포함된 함수는 처리하지 않음. 아쉬운 부분이지만 문제는 없음.
    if match:
        funcname = match.group(1)
        return funcname
    else:
        return None


def exec_with_timeout(code, timeout, globals_dict=None, locals_dict=None):
    import signal
    import threading
    from concurrent.futures import ThreadPoolExecutor, TimeoutError

    if globals_dict is None:
        globals_dict = {}
    if locals_dict is None:
        locals_dict = {}

    def run_code():
        exec(code, globals_dict, locals_dict)
        return locals_dict.get('__return_value__', None)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_code)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            raise TimeoutError("Execution timed out")


def _execute(code, code_return: str):
    exit_code = 0
    # these imports are for locals() (to provide `global() context` to exec()
    import itertools
    import math
    import random
    from fractions import Fraction

    import sympy
    import sympy as sp
    from sympy import Symbol
    from sympy import isprime as is_prime
    from sympy import symbols

    # pip installed olympiad, and marker to avoid frequent errors of math solving

    def _mock_input(prompt=""):
        """
        to ignore input() to be attempting user in the code
        """
        return ""

    try:
        locals_ = locals()
        locals_["input"] = _mock_input
        if (
            "import matplotlib" in code
            or "import matplotlib.pyplot" in code
            or "plt.figure" in code
        ):
            code = "import matplotlib\nmatplotlib.use('Agg')\n" + code

        exec(code, locals_)  # code로 local_ 딕셔너리 업데이트
        solution = locals_.get("solution", None)
        funcname = get_func_name_from_string(code)  # for nontrivial function names

        if solution is not None:
            ans = solution()  # this will use locals()
        elif funcname:  # if any function name appears
            new_code = "import math\n" + code + f"\nresult = {funcname}()"
            loc = {}
            locals__ = locals()
            locals__["input"] = _mock_input
            exec(new_code, locals__, loc)  # this will also use locals()

            ans = loc["result"]
        else:
            executed_code = (
                "import math\n"
                + "import datetime\n"
                + "\n".join([xx[4:] for xx in code.strip().split("\n")[1:-1]])
            )
            exec(executed_code, {"input": _mock_input}, locals())
            locals_ = locals()
            ans = locals_.get(code_return, None)

        # check if ans is sympy object so it needs to be converted by `sp.latex()`
        if isinstance(ans, sp.Basic):
            try:
                ans = sp.latex(ans)
            except Exception as e:
                exit_code = -2
                print(e)
                print(f"cannot be `sp.latex()`'d")
        return ans, exit_code

    except Exception as exp:
        # print("Executing code error", exp)
        # print(f"code:\n{code}")
        # print(f"{code_return=}")
        # print(f"{(solution is None)=}")
        # print(f"{funcname=}")
        return None, -1


### executing a code
def safe_execute_turbo(code_string: str):
    try:
        # === find code snippets between def solution(): and return ===
        code_list = code_string.strip().split("\n")

        new_code_list = []
        all_codes = []
        code_return = "ans"

        for i in range(len(code_list)):
            if code_list[i].startswith("import "):
                all_codes.append(code_list[i])
            if re.search(r"def (\w+)\(", code_list[i]) and code_list[i].startswith(
                "def "
            ):  # avoid including inner function definition
                new_code_list.append(code_list[i])
                for j in range(i + 1, len(code_list)):
                    if code_list[j].startswith("    "):
                        new_code_list.append(code_list[j])
                    if code_list[j].startswith(
                        "    return "
                    ):  # affirms outtermost return
                        code_return = code_list[j].split("return ")[1].strip()
                        break  # it could possibly miss the return if the function written with if-elif-else return at the end, which might be scarce.
                all_codes.append("\n".join(new_code_list))
                new_code_list = []

        if all_codes:
            new_code = "\n\n".join(
                all_codes
            )  # all_codes[-1] # if we parsed more than one function, we need to use them all.

            with math_util.timeout(seconds=3):
                ans, exit_code = _execute(new_code, code_return)
            if exit_code == 0:
                ans = _convert_to_float_if_possible(ans)
                ans = _convert_to_str_if_not_none_nor_float(ans)
            else:
                ans = None
        else:
            ans = None
    except:
        ans = None
    return ans


######################
## cot execution ##
######################


def _find_the_last_numbers(txt: str) -> str:
    # Regex pattern to match numbers with optional commas as thousand separators
    # and optional scientific notation
    pattern = r"[+\-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:[eE][+\-]?\d+)?"
    matches = re.findall(pattern, txt)

    if matches:
        # Replace commas to handle the number correctly as a standard numeric format
        return matches[-1].replace(",", "")
    else:
        return None


def _find_the_last_latex_expression(txt: str) -> Union[str, List]:
    latex_pattern = (
        r"(?:\\\[|\\\(|\$).+?(?:\\\]|\\\)|\$)"  # r'\\(?:\[|\().*?\\(?:\]|\))'
    )
    matches = re.findall(latex_pattern, txt, re.DOTALL)
    if matches:
        found = matches[-1]
    else:
        found = []
    return found


def extract_ans_from_cot_MATHnOCW(solution: str) -> str:
    """
    parsing symbolic or any form answer-string of interest from CoT result
    this is for parsing answers from cot solution of MATH/ocw_courses prompt
    see the corresponding prompt at `rims_minimal/src/utils/ocw_MATH_prompts.yaml`
    """

    prefix_list = [
        "Final answer:",
        "Final Answer:",
        "The final answer is",
        "The final answer",
        "### Final Answer",
        "### Final answer",
    ]
    
    last_solution = None
    for keyword, is_prefix in list(zip(prefix_list, [True] * len(prefix_list))) + [(". I hope it is correct.",), False]:
        
        if is_prefix:
            solution = solution.split(keyword)[-1].strip()
        else:
            solution = solution.split(keyword)[0].strip()
            
        found_latex = _find_the_last_latex_expression(solution)
        if found_latex:
            return found_latex
        else:
            found_numbers = _find_the_last_numbers(solution)
            if found_numbers:
                return found_numbers
            else:
                last_solution = found_numbers

    return last_solution


def extract_num_turbo(solution: str):
    """
    parsing (executing) cot result into a float
    see the prompt at `src/utils/math_prompt.py`
    This is for GSM prompt (from Automatic Model Selection Reasoning https://arxiv.org/pdf/2305.14333.pdf)
    """
    ans: str = solution.strip().replace("So the answer is ", "")
    prd: Union[str, None] = _find_the_last_numbers(ans)
    prd = float(prd.replace(",", "").rstrip(".")) if prd else prd

    return prd


#####################
### majority vote ###
#####################


def get_concordant_answer(
    answers: list,
    dataset_type: Literal["gsm", "svamp", "ocw", "math"] = "gsm",
):
    """
    check if there is a pair of concordant answers amongst 3 answer submission.
    input: [cot_ans, pal_ans, p2c_ans]
    output: ans if concordant else None

    *recommend to put answers in the order of cot going first (usually they are intgers)

    **for math and ocw, safe_execute_turbo polishes returned answer with `sp.latex` if the result is sympy object (See the last line of `def _execute`)
    """
    answers_no_none = [a for a in answers if (a is not None and a != "None")]

    if dataset_type in ["svamp", "gsm"]:
        answers_no_none = [_convert_to_float_if_possible(a) for a in answers_no_none]
        answers_no_none = [
            a for a in answers_no_none if isinstance(a, Union[float, int])
        ]
        if len(answers_no_none) == 0:
            res = None
        else:
            majority, count = Counter(answers_no_none).most_common(1)[0]
            if count >= 2:
                res = majority if isinstance(majority, float) else None
            else:
                # continue to check if 1e-3 tolerance same thing exist
                if len(answers_no_none) == 0:
                    res = None
                elif len(answers_no_none) == 1:
                    res = answers_no_none.pop()
                    return majority if isinstance(majority, float) else None
                elif len(answers_no_none) == 2:
                    if abs(answers_no_none[0] - answers_no_none[1]) < 1e-3:
                        res = answers_no_none[0]
                    else:
                        res = None
                else:
                    for a1, a2 in combinations(answers_no_none, 2):
                        if abs(a1 - a2) < 1e-3:
                            res = a1
                            break
                        else:
                            res = None
        return res
    elif dataset_type in ["math"]:
        answers_normalized = []
        for a in answers_no_none:
            try:
                with math_util.timeout(seconds=30):
                    res = math_util.normalize_final_answer(str(a))
                    answers_normalized.append(res)
            except TimeoutError as e:
                print(
                    "math_util.normalize_final_answer raise timeout error. Skip this answer."
                )
                pass
        print("answers_normalized len", len(answers_normalized))
        print("answers_normalized", answers_normalized)

        if len(answers_normalized) == 0:
            res = None
        elif len(answers_normalized) == 1:
            res = answers_no_none.pop()
        elif len(answers_normalized) == 2:
            cond = math_util.is_equiv(answers_normalized[0], answers_normalized[1])
            res = answers_no_none[0] if cond else None
        else:  # len()==3
            revert_normalized = dict(zip(answers_normalized, answers_no_none))

            for a1, a2 in combinations(answers_normalized, 2):
                cond = math_util.is_equiv(a1, a2)
                res = revert_normalized[a1] if cond else None
                if res is not None:
                    break
        return res
    elif dataset_type in ["ocw"]:
        if len(answers_no_none) == 0:
            res = None
        elif len(answers_no_none) == 1:
            res = answers_no_none.pop()
        elif len(answers_no_none) == 2:
            cond = math_util.is_equiv_ocw(answers_no_none[0], answers_no_none[1])
            res = answers_no_none[0] if cond else None
        else:  # len()==3
            for a1, a2 in combinations(answers_no_none, 2):
                cond = math_util.is_equiv_ocw(a1, a2)
                res = a1 if cond else None
                if res is not None:
                    break
        return res  # no concordant answers


def bucket_count_floating_numbers(numbers: List, tolerance: float = 1e-3) -> Counter:
    """
    used for `get_concordant_answer_n`

    # Example usage:
    numbers = [0.001, 0.002, 0.0025, 0.003, 1.000, 1.0005, 0.999]
    tolerance = 1e-3
    result = bucket_count_floating_numbers(numbers, tolerance)
    print(result)
    """

    # Function to bucket the numbers
    def bucket_number(num, tolerance):
        # 0.00251 will assign to 0.003, 0.0025 will assign to 0.002
        try:
            buck_num = round(num / tolerance) * tolerance
        except Exception as e:
            print(e)
            print("`bucket_count_floating_numbers()` fail!")
            buck_num = None
        return buck_num

    # Create a dictionary to count occurrences of each bucketed value
    bucket_counts = {}

    for number in numbers:
        # Bucket the number
        bucketed_number = bucket_number(number, tolerance)

        # Update the count for this bucket
        if bucketed_number is None:
            continue
        elif bucketed_number in bucket_counts:
            bucket_counts[bucketed_number] += 1
        else:
            bucket_counts[bucketed_number] = 1

    # Return the dictionary with counts of each bucketed value
    return Counter(bucket_counts)


def bucket_count_ocw_math_ans(answers: List[str], dataset_type: Literal["ocw", "math"]):
    """
    used for `get_concordant_answer_n`
    """
    answers_counts = Counter(answers)
    if len(answers_counts) == 1:
        pass
    else:
        eq_f = (
            math_util.math_check_answer
            if dataset_type == "math"
            else math_util.ocw_check_answer
        )
        for a1, a2 in combinations(answers_counts.keys(), 2):
            if eq_f(a1, a2):
                sumcount = answers_counts[a1] + answers_counts[a2]
                answers_counts[a1] = sumcount
                answers_counts[a2] = sumcount

    return answers_counts


def get_concordant_answer_n(
    answers: List, dataset_type: Literal["gsm", "svamp", "ocw", "math"] = "gsm"
) -> Union[float, str, List, None]:
    """
    n>3 version of the get_concordant_answer
    """
    answers_no_none = [a for a in answers if (a is not None and a != "None")]
    if not answers_no_none:
        return None

    if dataset_type in ["svamp", "gsm"]:
        # relatively, arithmetics has less need for tolerance comparison. So, we can use Counter first.
        majority, count = Counter(answers_no_none).most_common(1)[0]
        if count >= 2:
            return majority if isinstance(majority, float) else None
        # no majority: count = 0, 1
        else:
            # continue to check if 1e-3 tolerance same thing exist
            if len(answers_no_none) == 0:
                return None
            elif len(answers_no_none) == 1:
                majority = answers_no_none.pop()
                return majority if isinstance(majority, float) else None
            else:  # len(answers_no_none) >= 2:
                # count the floating numbers
                bucket_counts = bucket_count_floating_numbers(answers_no_none)
                # if there is a bucket with more than 1 number, return the bucket
                majority_, count_ = bucket_counts.most_common(1)[0]
                if count_ > 1:
                    return majority_
                else:
                    return None

    else:  # dataset_type in ["math", "ocw"]:
        if len(answers_no_none) == 0:
            return None
        elif len(answers_no_none) == 1:
            return answers_no_none.pop()
        else:  # len(answers_no_none) >= 2:
            # count the floating numbers
            bucket_counts = bucket_count_ocw_math_ans(
                answers_no_none, dataset_type=dataset_type
            )
            # if there is a bucket with more than 1 number, return the bucket
            majority_, count_ = bucket_counts.most_common(1)[-1]
            maj2_, count2_ = bucket_counts.most_common(2)[-1]

            majorities = [majority_, maj2_] if count_ == count2_ else [majority_]
            if count_ > 1:
                return majorities
            else:
                return None


###################
### rims ###
###################
def get_answer_rims(solution: str, ans: str = "", method: str = ""):
    from .text_parse_functions import parse_num_from_answer

    try:
        if method == "cot":
            pred = parse_num_from_answer(ans)
        elif method in ["pal", "p2c"]:
            pred = safe_execute_turbo(solution)
        else:
            if "return " in solution:
                pred = safe_execute_turbo(solution)
            else:
                pred = parse_num_from_answer(ans)

    except Exception as e:
        print(e)
        pred = None
    return pred


def process_rims_out_dict(parse_dd: dict) -> dict:
    """
    in:
        parsed_dict: contains fields that is directly related to the prompt response such as...
            Attempt 1: solution1 string
            Answer 1: solution1 answer (raw string)
            Mistakes: [solution 1,2,3,...'s mistake string]
            ...
    out:
        eval_friendly_d (dict): contains eval-friendly parsed fields
            good_solution: solution string at the last
            good_ans: correct answer executed above
            good_method: correct method abbreviation (e.g. cot)
            bad_ans: [list of wrong answers]
            bad_method: [list of wrong methods before the correct one]
            bad_solutions: [list of wrong solutions before the correct one]
            mistakes: [list of mistakes]
            hint: [list of hints]

    """

    from .text_parse_functions import parse_method2

    # dataset_type = parse_dd["dataset_type"]
    attempts_keys = sorted([k for k in parse_dd.keys() if "Attempt" in k])
    ans_keys = sorted([k for k in parse_dd.keys() if "Answer" in k])
    # method_keys = sorted([k for k in parse_dd.keys() if 'Method' in k])

    if (
        ans_keys and attempts_keys
    ):  # answer and solutions inside. additionally Method key is also in the parse_dd
        good_solution = parse_dd[attempts_keys[-1]] if attempts_keys else None
        did_reflect = 0
        if "Workaround Method" in parse_dd.keys() and parse_dd["Workaround Method"]:
            did_reflect += len(parse_dd["Workaround Method"])
            good_method = parse_method2(parse_dd["Workaround Method"][-1])
            bad_method = []
            if "Method" in parse_dd.keys():
                bad_method.append(parse_method2(parse_dd["Method"].pop()))
            if len(parse_dd["Workaround Method"]) > 1:
                bad_method += [
                    parse_method2(mstr) for mstr in parse_dd["Workaround Method"][:-1]
                ]

            # ans and solutions
            good_ans = parse_dd[ans_keys[-1]]
            bad_ans = [parse_dd[ak] for ak in ans_keys[:-1]]

            good_solution = parse_dd[attempts_keys[-1]]
            bad_solution = [parse_dd[atk] for atk in attempts_keys[:-1]]

        elif "Method" in parse_dd.keys() and parse_dd["Method"]:
            if "Mistakes" in parse_dd.keys():
                did_reflect += len(parse_dd["Mistakes"])
            good_method = parse_method2(parse_dd["Method"][-1])
            bad_method = [parse_method2(m) for m in parse_dd["Method"][:-1]]

            # ans and solutions
            good_ans = parse_dd[ans_keys[-1]]
            bad_ans = [parse_dd[ak] for ak in ans_keys[:-1]]

            good_solution = parse_dd[attempts_keys[-1]]
            bad_solution = [parse_dd[atk] for atk in attempts_keys[:-1]]

        else:  # solved at once
            good_method = []
            if "Method" in parse_dd.keys():
                good_method = parse_method2(parse_dd["Method"])
            bad_method = []

            good_ans = parse_dd[ans_keys[-1]]
            bad_ans = []

            good_solution = parse_dd[attempts_keys[-1]]
            bad_solution = []

    else:  # rims queried for evaluation only. no answer nor solutions.
        did_reflect = 0
        good_solution = None
        good_method = None
        good_ans = None
        bad_solution = []
        bad_ans = []
        bad_method = []

    mistakes = []
    hint = []
    if "Mistakes" in parse_dd.keys():
        mistakes = parse_dd["Mistakes"]
    if "Hint for a better Method choice" in parse_dd.keys():
        hint = parse_dd["Hint for a better Method choice"]

    if not len(bad_solution) == len(bad_ans) == len(bad_method):
        print(f"{bad_solution=}", f"{bad_ans=}", f"{bad_method=}")
        print(f"{good_solution=}", f"{good_ans=}", f"{good_method=}")
        print(f"{bad_solution=} possibly repetition generated (chatgpt, temp 0)")

    eval_friendly_d = dict(
        good_solution=good_solution,
        good_ans=get_answer_rims(good_solution, ans=good_ans, method=good_method),
        good_method=good_method,
        bad_solutions=bad_solution,
        bad_ans=[
            get_answer_rims(s, ans=a, method=m)
            for s, a, m in zip(bad_solution, bad_ans, bad_method)
        ],
        bad_method=bad_method,
        mistakes=mistakes,
        hint=hint,
        did_reflect=did_reflect,
    )
    return eval_friendly_d
