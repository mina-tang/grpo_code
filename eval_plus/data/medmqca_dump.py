from datasets import load_dataset
import json

dataset = load_dataset("openlifescienceai/medmcqa", split="train")

letter_map = {0: "a", 1: "b", 2: "c", 3: "d"}

with open("medmcqa_formatted.jsonl", "w") as f:
    for example in dataset:
        question_text = (
            f"{example['question']} "
            f"a) {example['opa']} b) {example['opb']} c) {example['opc']} d) {example['opd']}"
        )
        answer_letter = letter_map[example["cop"]]
        json.dump({"prompt": question_text, "answer": answer_letter}, f)
        f.write("\n")
