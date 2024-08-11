import re

from datasets import load_dataset


def extract(solution: str):
    def get_content_in_balanced_braces(answer: str):
        assert answer.startswith("{") and answer.endswith("}")

        stack = "{"
        count = 1
        for char in answer[1:]:
            stack += char
            if char == "{":
                count += 1
            elif char == "}":
                count -= 1
                if count == 0:
                    break

        return "".join(stack[1:-1])

    embraced_answer = re.search(r"\\boxed({.+})", solution.replace("\n", "")).group(1)
    return get_content_in_balanced_braces(embraced_answer)


dataset = load_dataset("hendrycks/competition_math")

num_proc = 2
test_data = dataset["test"]
# test_data = test_data.filter(
#     lambda example: example["type"] == "Prealgebra",
#     batched=False,
#     num_proc=num_proc,
# )
test_data = test_data.map(
    lambda example: {"answer": extract(example["solution"])},
    batched=False,
    num_proc=1,
)
test_data = test_data.rename_column("problem", "question")

test_data.to_json("MATH-full.jsonl", lines=True)
