train_files = {
    "original": "data/output/txt/ABC.train.gold.bea19.orig.txt",
    "corrected": "data/output/txt/ABC.train.gold.bea19.cor.txt",
}

test_files = {
    "original": "data/output/txt/ABCN.dev.gold.bea19.orig.txt",
    "corrected": "data/output/txt/ABCN.dev.gold.bea19.cor.txt",
}



# Add these configurations
dataset_train_filename = "data/output/dataset/train.jsonl"
dataset_val_filename = "data/output/dataset/val.jsonl"
dataset_test_filename = "data/output/dataset/test.jsonl"
dataset_test_result_baseline_filename = "data/output/result/test_result_baseline.jsonl"
dataset_test_result_finetuned_filename = "data/output/result/test_result_finetuned.jsonl"

train_rate = 0.8

# Job
run_id = "20250224"
file_id_filename = f"data/output/job/{run_id}/file_ids.json"
job_id_filename = f"data/output/job/{run_id}/job_id.json"

# Model
fine_tuning_base_model_id = "gpt-4o-2024-08-06"  # or your preferred base model
model_suffix = "grammar-correction"

inference_finetuned_model_temperature = 0

inference_base_model_id = "gpt-4o-2024-08-06"
inference_base_model_temperature = 0

DEFAULT_LOG_LEVEL = "INFO"
