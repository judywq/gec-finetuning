import json
from lib.utils import backup_output_file
import os

input_file = "data/output/dataset/test.jsonl"
# target_file = "data/output/result/test_result_deepseek_baseline.jsonl"
# target_file = "data/output/result/test_result_gpt_4o_baseline.jsonl"
target_file = "data/output/result/test_result_gpt_4o_finetuned.jsonl"


def check_duplicate_original(input_file: str):
    with open(input_file, "r") as f:
        lines = f.readlines()

    duplicate_count = 0
    text_set = set()
    for line in lines:
        data = json.loads(line)
        text = data["metadata"]["original"]
        text += data["metadata"]["corrected"]
        
        if text in text_set:
            print(f"Duplicate: {text}")
            duplicate_count += 1
        else:
            text_set.add(text)

    print(f"Duplicate count: {duplicate_count}")

def add_sentence_id_to_dataset(input_file: str, target_file: str):
    if not os.path.exists(target_file):
        print(f"File {target_file} does not exist")
        return

    backup_file = backup_output_file(target_file)
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    mapping = {}
    for line in lines:
        data = json.loads(line)
        metadata = data["metadata"]
        sentence_id = metadata.pop("sentence_id")
        
        sig = metadata["original"] + metadata["corrected"]
        mapping[sig] = sentence_id
        
    with open(backup_file, "r", encoding="utf-8") as f_in:
        with open(target_file, "w", encoding="utf-8") as f_out:
            for line in f_in.readlines():
                data = json.loads(line)
                target_metadata = data[2]
                sig = target_metadata["original"] + target_metadata["corrected"]
                sentence_id = mapping.get(sig, None)
                if sentence_id is None:
                    print(f"Sentence id not found for {sig}")
                else:
                    target_metadata["sentence_id"] = sentence_id
                f_out.write(json.dumps(data) + "\n")
    
    print(f"Added sentence id to {target_file}")

if __name__ == "__main__":
    # check_duplicate_original(input_file)
    add_sentence_id_to_dataset(input_file, target_file)
