# Prepare the data and finetune gpt-4o on openai

import os
from lib.dataset_preparation import DatasetPreparation
import settings
from lib.utils import setup_log


def main():
    print(os.environ["OPENAI_API_KEY"])
    # Create output directory if it doesn't exist
    os.makedirs("data/output/dataset", exist_ok=True)
    
    # Prepare the dataset
    dataset_prep = DatasetPreparation(settings)
    dataset_prep.run()
    

if __name__ == "__main__":
    setup_log()
    main()
