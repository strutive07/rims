from pathlib import Path
from typing import Literal

import yaml
from query_obj import BaseQueryObject

THIS_PARENT = Path(__file__).parent.resolve()


class P2CQueryObject(BaseQueryObject):
    def __init__(
        self,
        dataset_type: str = "gsm",  # ocw math
        plan_temperature: float = 0.0,
    ):
        self.dataset_type = dataset_type
        self.plan_temperature = plan_temperature

        self.plan = None
        self.code_query = None
        self.plan_query = None

    async def async_query(
        self,
        question: str,
        temperature: float = 0.0,
        backbone: str = "chatgpt",
        n: int = 1,
        seed: int = 777,
        max_tokens: int = 2048,
        stop="\n\n\n",
    ):
        contents, query_message, resp, meta = await super().async_query(
            question, temperature, backbone, n, seed, max_tokens, stop
        )

        meta["plan"] = self.plan
        meta["codequery"] = self.code_query
        meta["planquery"] = self.plan_query

        return contents, query_message, resp, meta

    async def prepare_query(self, question: str, backbone: str, **kwargs):
        seed = kwargs["seed"]
        model_name = self.backbone2model(backbone)

        k_fewshot = 8  # default init 8, for openLLM cases
        if backbone.startswith("gpt4"):
            # print(f'gpt-4 uses k_fewshot=5 as default (p2c fs_prompting)')
            k_fewshot = 5
        elif backbone.startswith("chatgpt"):  # ("gpt-3.5-turbo-16k-0613"):
            # print(f'gpt-3.5 uses k_fewshot=8 as default (p2c fs-prompting)')
            k_fewshot = 8

        plan_query_msg = get_plan_prompt(question, k_fewshot=k_fewshot)

        plan_max_tokens_d = {
            "gsm": 800,
            "ocw": 800,
            "math": 800,
            "svamp": 800,
        }

        resp = await self.async_query_to_llm(
            model=model_name,
            max_tokens=plan_max_tokens_d[self.dataset_type],
            stop="Question: ",
            messages=plan_query_msg,
            temperature=self.plan_temperature,
            top_p=1.0,
            n=1,  # plan*1 + code*n (bad for p2c acc but perf. consideration)
            seed=seed,
        )

        contents = self.get_contents(resp)
        plan = [self.postprocess_plan(content) for content in contents][0]

        if not plan:
            return -1

        code_query_msg = get_plan2code_prompt(question, plan=plan, k_fewshot=k_fewshot)
        self.plan = plan
        self.code_query = code_query_msg
        self.plan_query = plan_query_msg
        return code_query_msg

    def query_error_msg(self, query_message):
        if query_message == -1:
            return (
                True,
                (None, None, {"codequery": None, "planquery": plan_query_msg}),
            )
        else:
            return False, None

    @staticmethod
    def postprocess_plan(rawanswer: str):
        # lines = [l for l in rawanswer.split('\n') if '</end>' not in l]
        lines = rawanswer.split("\n")
        if len(lines) >= 1:
            plan_ = "\n".join(lines)
        else:
            print("plan gen failed")
            print(f"{rawanswer=}")
            plan_ = ""
        return plan_

    @staticmethod
    def postprocess_code(rawanswer: str, k_fewshot: int = 0):
        def remove_prints(code: str) -> str:
            lines = code.split("\n")
            lines_ = [
                l if not l.startswith("print(") else l.replace("print(", "# print(")
                for l in lines
            ]
            code_ = "\n".join(lines_)
            return code_

        try:
            # 1 removing starting wrap ```
            if "```python" in rawanswer:
                rawanswer = rawanswer.split("```python")[-1]
            elif rawanswer.startswith("```"):
                rawanswer = rawanswer.split("```")[-1]

            # 2 removing ``` at the end
            code = rawanswer.split("```")[0]  # ending ``` removal

            code = remove_prints(code)
            assert code
        except:
            print("code gen fails (unexecutable or funcname?)")
            print(f"code:\n{rawanswer}")
            code = ""
        return code


def get_plan_prompt(question: str, k_fewshot: int = 0) -> str:
    """
    prep prompt for plan generation
    # put "Question: " in front of the `question`

    """
    PLAN_F = THIS_PARENT / "new_p2c_plan_prompts.yaml"  # "prompts_plan_v2.yaml"
    PLAN_PROMPTS_D = yaml.full_load(open(PLAN_F))
    prompt_d = PLAN_PROMPTS_D

    # q = data['question']
    q = question
    system = prompt_d["system_msg"]
    user_tmp = prompt_d["user_template"]
    user_attempt = user_tmp.replace("{QUESTION}", q)  # f"Question: {q}")

    fewshots_user = prompt_d["fewshots_user"][
        :k_fewshot
    ]  # list of fewshot strings include Question: as a stop sequence.
    fewshots_assistant = prompt_d["fewshots_assistant"][:k_fewshot]

    msgs = [
        {"role": "system", "content": system},
    ]
    for fu, fa in zip(fewshots_user, fewshots_assistant):
        usr = {"role": "user", "content": fu}
        astnt = {"role": "assistant", "content": fa}
        msgs.append(usr)
        msgs.append(astnt)
    msgs.append({"role": "user", "content": user_attempt})

    return msgs


def get_plan2code_prompt(
    question: str,  # data:dict,
    plan: str = "",
    k_fewshot: int = 0,
    custom_idxs: list = None,
):
    """
    prep prompt for code generation
    """
    CODE_F = THIS_PARENT / "new_p2c_code_prompts.yaml"  # "prompts_code_v2.yaml"
    prompt_d = yaml.full_load(open(CODE_F))

    q = question  # data['question']
    system = prompt_d["system_msg"]
    user_tmp = prompt_d["user_template"]
    plan_with_tabs = plan.replace("\n", "\n" + " " * 4)
    user_attempt = user_tmp.replace("{PLAN}", plan_with_tabs).replace("{QUESTION}", q)

    if not custom_idxs:
        fewshots_user = prompt_d["fewshots_user"][
            :k_fewshot
        ]  # list of fewshot strings include Question: as a stop sequence.
        fewshots_assistant = prompt_d["fewshots_assistant"][:k_fewshot]
    else:
        fewshots_user = [prompt_d["fewshots_user"][i] for i in custom_idxs]
        fewshots_assistant = [prompt_d["fewshots_assistant"][i] for i in custom_idxs]

    msgs = [
        {"role": "system", "content": system},
    ]
    for fu, fa in zip(fewshots_user, fewshots_assistant):
        usr = {"role": "user", "content": fu}
        astnt = {"role": "assistant", "content": fa}
        msgs.append(usr)
        msgs.append(astnt)
    msgs.append({"role": "user", "content": user_attempt})

    return msgs
