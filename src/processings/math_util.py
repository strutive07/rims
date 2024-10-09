import logging
import re
import signal

import numpy as np

try:
    import sympy
    from sympy.core.sympify import SympifyError
    from sympy.parsing.latex import parse_latex
except ModuleNotFoundError:
    raise Exception(
        "`sympy` is required for generating translation task prompt templates. \
please install sympy via pip install lm-eval[math] or pip install -e .[math]",
    )

from typing import Any, Callable, List, Optional

# ==== https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/minerva_math/utils.py ====


### high-level functions ###
def gsm_check_answer(a1, a2):
    try:
        a1, a2 = map(float, [a1, a2])
        decision = abs(a1 - a2) < 1e-3
    except Exception as e:
        print(e)
        print(f"{a1=}, {a2=}")
        decision = None
    return decision


def ocw_check_answer(a1, a2):
    """
    check if a1 and a2 are equivalent in ocw
    """
    try:
        a1, a2 = map(str, [a1, a2])
        print(f"a1 {a1}")
        print(f"a2 {a2}")
        decision = is_equiv_ocw(a1, a2)
    except Exception as e:
        print(e)
        decision = False
    print(f"decision {decision}")
    return decision


def math_check_answer(a1, a2):
    """
    check if a1 and a2 are equivalent in math
    """
    try:
        with timeout(seconds=5):
            a1, a2 = map(str, [a1, a2])
            decision = is_equiv(normalize_final_answer(a1), normalize_final_answer(a2))
    except Exception as e:
        print(e)
        decision = False
    return decision


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


### atomic functions ###
def is_equiv(x1: str, x2: str) -> bool:
    """
    x1 and x2 are normalized latex string
    """
    try:
        with timeout(seconds=3):
            # before relying on parse_latex, which is problematic, try exact match by string first
            # added by seonil
            if x1.replace(" ", "") == x2.replace(" ", ""):
                return True
            else:
                pass

            # original code: parse and then sympy-equivalence by diff
            try:
                parsed_x1 = parse_latex(x1)
                parsed_x2 = parse_latex(x2)
            except (
                sympy.parsing.latex.errors.LaTeXParsingError,
                sympy.SympifyError,
                TypeError,
            ):
                logging.debug(f"couldn't parse one of {x1} or {x2}")
                return False

            try:
                diff = parsed_x1 - parsed_x2
            except TypeError:
                logging.debug(f"couldn't subtract {x1} and {x2}")
                return False

            try:
                if sympy.simplify(diff) == 0:
                    return True
                else:
                    return False
            except ValueError:
                logging.debug(
                    f"Had some trouble simplifying when comparing {x1} and {x2}"
                )
    except TimeoutError:
        logging.debug(f"Timed out comparing {x1} and {x2}")
        return False
    except ImportError as e:
        logging.error(e)
        raise
    except Exception as e:
        logging.debug(f"Failed comparing {x1} and {x2} with {e}")
        return False


# these constants also used in OCW math below. do not detach this from here.
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS = [
    "\\left",  # added by seonil
    "\\right",  # added by seonil
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]

def remove_end_bracket_pure_numeric(text):
    pattern = r'^(\d+\.?\d*)\]$'
    match = re.match(pattern, text)
    if match:
        return match.group(1)
    return text

def normalize_final_answer(final_answer: str) -> str:
    # https://github.com/wellecks/lm-evaluation-harness/blob/bec2172e72be4adc70e85957cc97a2fbe70c207b/lm_eval/mixins.py#L188
    # original function name is `normalize_tex`
    """
    Normalize a final answer to a quantitative reasoning question.

    Copied character for character from appendix D of Lewkowycz et al. (2022)
    """

    final_answer = final_answer.replace('\frac', '\dfrac')
    final_answer = final_answer.replace('\n', '')
    final_answer = final_answer.strip()

    while(True):
        if final_answer.startswith('\(') and final_answer.endswith('\)'):
            final_answer = final_answer[2:-2].strip()
        else:
            break
            
    while(True):
        if final_answer.startswith('\[') and final_answer.endswith('\]'):
            final_answer = final_answer[2:-2].strip()
        else:
            break

    final_answer = remove_end_bracket_pure_numeric(final_answer)

    
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer


# ==== for OCW (from minerva appendix) ====
INVALID_ANSWER = "[invalidanswer]"


def is_equiv_ocw(x1: str, x2: str, use_sym_exp_normalizer: bool = False) -> bool:
    """
    code took from Minerva original repository and adjusted for our use

    see OCWCourses::process_results
    https://github.com/wellecks/lm-evaluation-harness/blob/bec2172e72be4adc70e85957cc97a2fbe70c207b/lm_eval/tasks/ocw_courses.py#L153

    expects x1 and x2 to be string (latex)
    """
    if use_sym_exp_normalizer:
        raise ValueError(
            "`use_sym_exp_normalizer=True` is considered unreliable (tested, looks more suspicious about its evaluation \
                         \
                         \n\nsee `src/prompt_construction_src/tests/test_diff_by_parsing_cot.ipynb`"
        )

    # ensure x1, x2 be strings
    x1 = str(x1)
    x2 = str(x2)
    try:
        # original code checks if the reference answer is float() castable, but my case, x1, x2 are not certainly numberic or numeric with units --> normalize_numeric() to remove units and then float cast
        normalize_fn = normalize_numeric
        _is_equiv = numeric_equality_ocw
        float(normalize_fn(x1))
        float(normalize_fn(x2))
        # answer_type = "numeric"
    except ValueError as ve:
        if "=" in x1 or "=" in x2:
            normalize_fn = normalize_symbolic_equation
            _is_equiv = lambda x, y: x == y
            # answer_type = "equation"
        else:
            normalize_fn = (
                normalize_symbolic_expression
                if use_sym_exp_normalizer
                else normalize_final_answer
            )
            _is_equiv = is_tex_equiv
            # answer_type = "expression"

    if INVALID_ANSWER in (x1, x2):
        return False
    # x1, x2 = map(normalize_fn, [x1,x2])
    # print(x1)
    # print(x2)
    try:
        return _is_equiv(normalize_fn(x1), normalize_fn(x2))
    except Exception as e:
        print(e)
        return False


def numeric_equality_ocw(n1, n2, threshold=0.01):
    """
    from appendix of the Minerva paper
    """
    try:
        with timeout(seconds=1):
            # this assumes n1, n2 are numerics. so cast it into float
            n1, n2 = map(float, [n1, n2])
            if n1 is None or n2 is None:
                return False
            if "None" in [n1, n2]:
                return False
            if n1 == n2:  # exact match covered here
                return n1 == n2
            if np.isclose(n1, 0) or np.isclose(n2, 0) or np.isclose(n1 - n2, 0):
                return (
                    np.abs(n1 - n2) < threshold * np.abs(n1 + n2) / 2
                )  # original code cannot cover negative numbers, so added np.abs to threshold condition to cover it.
            else:
                return np.isclose(n1, n2)
    except Exception as e:
        print(e)
        return False


def normalize_symbolic_equation(s: Optional[str]):
    if not isinstance(s, str):
        return INVALID_ANSWER
    if s == INVALID_ANSWER:
        return INVALID_ANSWER
    if s.startswith("\\["):
        s = s[2:]
    if s.endswith("\\]"):
        s = s[:-2]
    s = s.replace("\\left(", "(")
    s = s.replace("\\right)", ")")
    s = s.replace("\\\\", "\\")
    if s.startswith("$") or s.endswith("$"):
        s = s.strip("$")
    try:
        maybe_expression = parse_latex(s)
        if not isinstance(maybe_expression, sympy.core.relational.Equality):
            # we have equation, not expression
            return INVALID_ANSWER
        else:
            return maybe_expression
    except:
        return INVALID_ANSWER


def normalize_symbolic_expression(s: Optional[str]):
    if not isinstance(s, str):
        return INVALID_ANSWER
    if s.startswith("\\["):
        s = s[2:]
    if s.endswith("\\]"):
        s = s[:-2]
    s = s.replace("\\left(", "(")
    s = s.replace("\\right)", ")")
    s = s.replace("\\\\", "\\")
    if s.startswith("$") or s.endswith("$"):
        s = s.strip("$")
    try:
        maybe_expression = parse_latex(s)
        if isinstance(maybe_expression, sympy.core.relational.Equality):
            # we have equation, not expression
            return INVALID_ANSWER
        if isinstance(maybe_expression, sympy.logic.boolalg.BooleanFalse):
            return INVALID_ANSWER
        else:
            return maybe_expression
    except Exception as e:
        print(e)
        return INVALID_ANSWER


def is_exp_equiv(x1: sympy.Basic, x2: sympy.Basic, time_limit=5) -> bool:
    """
    Determines whether two sympy expressions are equal.
    """
    if not str(x1) or not str(x2):
        return False
    if "nan" in [str(x1), str(x2)]:
        return False
    try:
        with timeout(seconds=time_limit):
            try:
                diff = x1 - x2
            except (SympifyError, ValueError, TypeError) as e:
                print(f"Couldn't subtract {x1} and {x2} with exception {e}")
                return False

            try:
                if sympy.simplify(diff) == 0:
                    return True
                else:
                    return False
            except (SympifyError, ValueError, TypeError) as e:
                print(f"Failed to simplify {x1}-{x2} with {e}")
                return False
    except TimeoutError as e:
        print(f"Timed out comparing {x1} and {x2}")
        return False
    except Exception as e:
        print(f"failed on unrecognized exception {e}")
        return False


def is_tex_equiv(x1: str, x2: str, time_limit=5) -> bool:
    """
    Determines whether two (ideally normalized using `normalize_text`) TeX expressions are equal.

    Does so by first checking for string exact-match, then falls back on sympy-equivalence,
    following the (Lewkowycz et al. 2022) methodology.
    """
    if not str(x1) or not str(x2):  # added
        return False
    if "nan" in [str(x1), str(x2)]:  # added
        return False
    if x1 == x2:
        # don't resort to sympy if we have full string match, post-normalization
        return True

    parsed_x2 = parse_tex(x2)
    # if not parsed_x2: # this line invokes error (Some sympy objects are not boolean-decisive)
    #     # if our reference fails to parse into a Sympy object,
    #     # we forgo parsing + checking our generated answer.
    #     return False

    return is_exp_equiv(parse_tex(x1), parsed_x2, time_limit=time_limit)


def parse_tex(text: str, time_limit: int = 5) -> sympy.Basic:
    """
    Wrapper around `sympy.parse_text` that outputs a SymPy expression.
    Typically, you want to apply `normalize_text` as a preprocessing step.
    """
    try:
        with timeout(seconds=time_limit):
            parsed = parse_latex(text)
    except (
        # general error handling: there is a long tail of possible sympy/other
        # errors we would like to catch
        Exception
    ) as e:
        print(f"failed to parse {text} with exception {e}")
        return None

    return parsed


def normalize_numeric(s):
    if s is None:
        return None
    for unit in [
        "eV",
        " \\mathrm{~kg} \\cdot \\mathrm{m} / \\mathrm{s}",
        " kg m/s",
        "kg*m/s",
        "kg",
        "m/s",
        "m / s",
        "m s^{-1}",
        "\\text{ m/s}",
        " \\mathrm{m/s}",
        " \\text{ m/s}",
        "g/mole",
        "g/mol",
        "\\mathrm{~g}",
        "\\mathrm{~g} / \\mathrm{mol}",
        "W",
        "erg/s",
        "years",
        "year",
        "cm",
    ]:
        s = s.replace(unit, "")
        s = s.strip()
    for maybe_unit in ["m", "s", "cm"]:
        s = s.replace("\\mathrm{" + maybe_unit + "}", "")
        s = s.replace("\\mathrm{~" + maybe_unit + "}", "")
        s = s.strip()
    s = s.strip("$")
    try:
        return float(eval(s))
    except:
        try:
            expr = parse_latex(s)
            if expr.is_number:
                return float(expr)
            return INVALID_ANSWER
        except:
            return INVALID_ANSWER


#### test answer latex parsing ### (test_*.py's)


def ocw_parse(x1: str, use_old: bool = False) -> str:
    """
    test ocw answer validity after parsing
    """
    x1 = str(x1)
    try:
        parser_f = normalize_numeric
        x1 = parser_f(x1)
        # float(x1)
    except Exception as e:
        if "=" in x1:
            parser_f = normalize_symbolic_equation
        else:
            parser_f = (
                normalize_final_answer if use_old else normalize_symbolic_expression
            )
        try:
            x1 = parser_f(x1)
        except Exception as e:
            x1 = f"PARSE_FAIL! {x1}, {str(e)}"
    return x1


def math_parse(x1: str) -> str:
    """
    test parsed math's answer's validity
    (do the same thing as in is_equiv)
    """
    try:
        parsed_x1 = parse_latex(x1)
    except Exception as e:
        parsed_x1 = "PARSE_FAIL! " + str(e)

    return str(parsed_x1)
