"""
code for model selection (i.e. simple-greedy)
"""
from pathlib import Path
from typing import Any, Dict, List, Literal

from query_obj import BaseQueryObject


class CrossAndMixQueryObject(BaseQueryObject):
    def __init__(self, dataset_type):
        self.dataset_type = dataset_type

    async def prepare_query(
        self,
        question: str,
        cot_pal_p2c_sln_d: Dict,
        **kwargs,  # to avoid overiding base class
    ):
        return get_select_prompt2(
            question,
            cot_pal_p2c_sln_d=cot_pal_p2c_sln_d,
            dataset_type=self.dataset_type,
        )

    # need override
    async def async_query(
        self,
        question: str,
        temperature: float = 0.0,
        backbone: str = "chatgpt",
        cot_pal_p2c_sln_d: Dict = None,
        n: int = 1,
        seed: int = 777,
        max_tokens: int = 4096,
        stop="\n\n\n",
    ):
        meta = {
            "method_obj": self.__class__.__name__,
            "dataset_type": getattr(self, "dataset_type", "not given"),
            "query_kwargs": {
                "backbone": backbone,
                "temperature": temperature,
                "n": n,
                "seed": seed,
                "max_tokens": max_tokens,
                "stop": stop,
            },
        }

        prepare_query_task = self.prepare_query(
            question,
            backbone=backbone,
            cot_pal_p2c_sln_d=cot_pal_p2c_sln_d,
            **{
                "temperature": temperature,
                "n": n,
                "seed": seed,
                "max_tokens": max_tokens,
                "stop": stop,
            },
        )
        query_message = await prepare_query_task
        is_error, error_msg = self.query_error_msg(query_message)
        if is_error:
            return error_msg

        model_name = self.backbone2model(backbone)

        call_llm_task = self.async_query_to_llm(
            model=model_name,
            max_tokens=max_tokens,
            stop=stop,
            messages=query_message,
            temperature=temperature,
            top_p=1.0,
            seed=seed,
            n=n,
        )
        resp = await call_llm_task
        contents = self.get_contents(resp)
        return contents, query_message, resp, meta

    def query_error_msg(self, query_message):
        return False, None


### getting prompts for each method ###
def get_select_prompt2(
    question: str,
    cot_pal_p2c_sln_d: dict = None,
    dataset_type: Literal["gsm", "svamp", "ocw", "math"] = None,
) -> List[Dict[str, str]]:
    # open up prompt template yaml file
    THIS_PARENT = Path(__file__).parent.resolve()
    prompt_yml = THIS_PARENT / "cross_and_mix_prompts.yaml"
    import yaml

    prompt_d: Dict[str, Any] = yaml.full_load(open(prompt_yml))

    select_prompt_key = f"{dataset_type}_select"
    ds_prom_d: Dict[str, Any] = prompt_d[select_prompt_key]

    # process with the question/solutions of interest to result in gpt_messages
    system: str = ds_prom_d["system"]
    msgs: List[Dict[str, str]] = ds_prom_d["contexts"]

    # make user's query from (question, cot_pal_p2c_sln_d)
    q = question
    to_replace_keys = "{COT_SOLUTION} {PAL_SOLUTION} {P2C_SOLUTION}"

    user_tmp: str = ds_prom_d["user_tmp"]

    user_tmp = user_tmp.replace("{QUESTION}", q)

    for to_replace, to_be in zip(to_replace_keys.split(), cot_pal_p2c_sln_d.values()):
        user_tmp = user_tmp.replace(to_replace, to_be)
    user_attempt = user_tmp

    msgs.append({"role": "user", "content": user_attempt})

    return msgs
