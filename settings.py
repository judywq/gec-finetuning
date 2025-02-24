train_files = {
    "original": "data/output/txt/ABC.train.gold.bea19.orig.txt",
    "corrected": "data/output/txt/ABC.train.gold.bea19.cor.txt",
}

val_files = {
    "original": "data/output/txt/ABCN.dev.gold.bea19.orig.txt",
    "corrected": "data/output/txt/ABCN.dev.gold.bea19.cor.txt",
}

# Add these configurations
dataset_train_filename = "data/output/dataset/train.jsonl"
dataset_val_filename = "data/output/dataset/val.jsonl"

# Job
run_id = "20250224"
file_id_filename = f"data/output/job/{run_id}/file_ids.json"
job_id_filename = f"data/output/job/{run_id}/job_id.json"

# Model
fine_tuning_base_model_id = "gpt-4o-2024-08-06"  # or your preferred base model
model_suffix = "grammar-correction"

DEFAULT_LOG_LEVEL = "DEBUG"
