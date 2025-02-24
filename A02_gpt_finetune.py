# Prepare the data and finetune gpt-4o on openai

import os
from lib.finetuning_helper import FineTuningHelper
from lib.utils import setup_log
import settings

skip_if_exist = True
# skip_if_exist = False

def main():
    # Create output directory if it doesn't exist
    os.makedirs(f"data/output/job/{settings.run_id}", exist_ok=True)
    
    # Run fine-tuning
    finetuning = FineTuningHelper(settings)
    finetuning.run(wait_for_job=True, skip_if_exist=skip_if_exist)

if __name__ == "__main__":
    setup_log()
    main()
