from datasets import load_dataset
import json
# For MCQ
SYSTEM_PROMPT = """You can add explanations or comments before or after, \
but the text inside the <answer> tags must contain only the final MCQ answer exactly \
as one of the provided options (e.g., "d) Roux en Y Duodenal By pass"). \
For example:  \
Here is my reasoning... <answer> d) Roux en Y Duodenal By pass <answer> \
Do not include anything else inside the <answer> tags. \
"""

dataset = load_dataset("openlifescienceai/medmcqa", split="train")

letter_map = {0: "a", 1: "b", 2: "c", 3: "d"}

with open("medmcqa_formatted.jsonl", "w") as f:
    for example in dataset:
        question_text = (
            f"{example['question']} "
            f"a) {example['opa']} b) {example['opb']} c) {example['opc']} d) {example['opd']}"
        )
        answer_letter = letter_map[example["cop"]]
        json.dump({"prompt": question_text + "\n\n" + SYSTEM_PROMPT, "answer": answer_letter}, f)
        f.write("\n")
