from datasets import Dataset, DatasetDict
import os
import settings

train_file_original = settings.train_files["original"]
train_file_corrected = settings.train_files["corrected"]

test_file_original = settings.test_files["original"]
test_file_corrected = settings.test_files["corrected"]


def compose_dict(original_file, corrected_file):    
    # Read the files
    with open(original_file, 'r', encoding='utf-8') as f1, open(corrected_file, 'r', encoding='utf-8') as f2:
        original_lines = f1.read().splitlines()
        corrected_lines = f2.read().splitlines()

    # Create a dictionary with your data
    data = {
        'original': original_lines,
        'corrected': corrected_lines
    }
    return data

train_data = compose_dict(train_file_original, train_file_corrected)
train_dataset = Dataset.from_dict(train_data)

test_data = compose_dict(test_file_original, test_file_corrected)
test_dataset = Dataset.from_dict(test_data)

dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

print(dataset)

# Save the dataset
# os.makedirs(settings.custom_finetuning_output_dir, exist_ok=True)
# dataset.save_to_disk(settings.custom_train_file)

# Upload the dataset to the hub
dataset.push_to_hub("judywq/gec-dataset", private=False)
