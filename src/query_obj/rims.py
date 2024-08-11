import re

from query_obj import BaseQueryObject


# bear me
# : got this from legacy for compatibility to rims prompt files...
class PromptStr(str):
    def __init__(self, template_str):
        super().__init__()
        self += template_str

    def sub(self, placeholder_name: str, tobe: str):
        return PromptStr(
            self.replace(f"[{placeholder_name}]", str(tobe))
        )  # it always returns copy substituted content strings

    def sub_map(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.get_placeholder_names():
                self = self.sub(k, v)
        return self

    def get_placeholder_names(self) -> list:
        return re.findall(r"\[(.*?)\]", self)


class RimsQueryObject(BaseQueryObject):
    def __init__(
        self,
        #  dataset_type:Literal["gsm", "ocw", "math"]=None,
        prompt_tmp: str = None,
    ):
        # self.dataset_type = dataset_type
        self.prompt_tmp = PromptStr(prompt_tmp)

    async def prepare_query(self, question: str, backbone: str, **kwargs):
        return get_rims_prompt(
            question=question,
            backbone=backbone,  # this is never used below, but for BaseQueryObject.async_query
            prompt_tmp=self.prompt_tmp,
            # dataset_type=self.dataset_type,
        )

    def query_error_msg(self, query_message):
        return False, None


def get_rims_prompt(
    prompt_tmp: PromptStr = None,  # path
    question: str = None,
    backbone: str = None,
    # dataset_type: Literal["gsm", "ocw", "math"] = None,
):
    """
    This function is used to generate the CoT prompt.
    append "Question: " to the `question`
    """

    prompt = prompt_tmp.sub("QUESTION", question)  # sub always returns copy of itself
    assert isinstance(prompt, str)
    messages = [{"role": "user", "content": prompt}]

    return messages
