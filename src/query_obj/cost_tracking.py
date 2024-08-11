from functools import update_wrapper
from typing import Any, Dict


class CountTokens:
    """
    # llm_query_utils.py
    @CountTokens
    def query_cot():
        return

    # main.py
    ...
    ..
    .

    query_cot.print_summary()


    will initialize a CountTokens object with the query_cot function as the `self.func` attribute, but letting function profile as `query_cot`

    This decorator will count the total number of tokens used by the function
    """

    def __init__(self, func):
        self.func = func
        self.n_called = 0
        # for cost tracking
        self.n_sample = 0  # *.completion.create(n:int=)
        self.total_toks_in = 0
        self.total_toks_out = 0
        # for context length estimation
        self.max_toks_in = 0
        self.max_toks_out = 0
        update_wrapper(
            self, func
        )  # Update the wrapper function (self) to look like func

    def tok_info_from_query_funcs(self, query_returns: Any = None) -> tuple:
        """
        from `query_returns` of query functions (query_cot|pal|plancode|selection...etc), get the total_number of the
        """

        # if the query func does not return multiple variables as a tuple, then wrap it in a tuple
        if not isinstance(query_returns, tuple):
            query_returns = (query_returns,)

        # CompletionUsage(completion_tokens:int, prompt_tokens:int, total_tokens:int)
        response = query_returns[-1]
        completion_usage = response.usage
        n: int = len(response.choices)

        return completion_usage, n

    def __call__(self, *args, **kwargs):
        # Call the original function
        results = self.func(*args, **kwargs)

        completion_usage, n = self.tok_info_from_query_funcs(results)
        toks_in, toks_out = (
            completion_usage.prompt_tokens,
            completion_usage.completion_tokens,
        )

        # update counts
        self.n_called += 1
        self.n_sample = max(n, self.n_sample)
        self.max_toks_in = max(self.max_toks_in, toks_in)
        self.max_toks_out = max(self.max_toks_out, toks_out / n)
        self.total_toks_in += toks_in
        self.total_toks_out += toks_out

        return results  # return the self.func.__call__() results

    def print_summary(self):
        print(f"Function: {self.func.__name__}")
        print(f"Max tokens in: {self.max_toks_in}")
        print(f"Max tokens out: {self.max_toks_out:.1f}")
        print(f"n_sample: {self.n_sample}")
        print(f"Total tokens in: {self.total_toks_in}")
        print(f"Total tokens out: {self.total_toks_out}")
        print(f"Number of calls: {self.n_called}")

    def tokens2usd(self, model: str = "") -> float:
        """
        # Example usage
        cost = tokens2usd(toks_in=500_000, toks_out=500_000, model="gpt-3.5-turbo-1106")
        print(f"Cost: ${cost:.4f}")
        """
        # Define the cost per 1,000,000 tokens for each model type for input and output
        pricing = {
            "gpt-3.5-turbo-1106": {"input": 1.00, "output": 2.00},
            "laba-gpt-35-turbo-1106": {"input": 1.00, "output": 2.00},
            "gpt-3.5-turbo-0613": {"input": 1.50, "output": 2.00},
            "GPT-35": {"input": 1.50, "output": 2.00},
            "gpt-3.5-turbo-16k-0613": {"input": 3.00, "output": 4.00},
            "laba-gpt-35-turbo-16k-0613": {"input": 3.00, "output": 4.00},
            "gpt-3.5-turbo-0301": {"input": 1.50, "output": 2.00},
            "gpt-3.5-turbo-0125": {"input": 0.50, "output": 1.50},
            "laba-gpt-35-turbo-0125": {"input": 0.50, "output": 1.50},
            "gpt-3.5-turbo-instruct": {"input": 1.50, "output": 2.00},
            "gpt-4-1106-preview": {"input": 10.00, "output": 30.00},
            "GPT4-1106": {"input": 10.00, "output": 30.00},
            "gpt-4-0125-preview": {"input": 10.00, "output": 30.00},
            "gpt-4": {"input": 30.00, "output": 60.00},
            "gpt-4-32k": {"input": 60.00, "output": 120.00},
        }

        # Check if the provided model is in the pricing dictionary
        if model not in pricing:
            raise ValueError(f"Unknown model type: {model}")

        # Calculate the cost in USD for input and output tokens separately
        cost_in_usd = (self.total_toks_in / 1_000_000) * pricing[model]["input"] + (
            self.total_toks_out / 1_000_000
        ) * pricing[model]["output"]

        print(
            f"{self.func.__name__}: \n{cost_in_usd:.2f} usd -- ({self.n_called} calls)"
        )

        return cost_in_usd
