import json
import os
import logging
from typing import List, Dict

from lib.io import save_to_jsonl, read_jsonl
from lib.finetuning_helper import FineTuningHelper
import settings


logger = logging.getLogger(__name__)

def main():
    input_file = settings.dataset_test_filename
    openai_batch_file = settings.dataset_test_openai_batch_filename
    # Read input data
    input_data = read_jsonl(input_file)
    
    model = get_finetuned_model()
    temperature = settings.inference_finetuned_model_temperature
    
    # Format data for batch processing
    formatted_data = format_for_batch(input_data, model, temperature)
    
    # Write formatted data to file
    save_to_jsonl(formatted_data, openai_batch_file)
    print(f"Batch file prepared: {openai_batch_file}")
    


def format_for_batch(data: List[Dict], model: str, temperature: float = 0) -> List[Dict]:
    """Format data for OpenAI batch processing."""
    formatted_data = []
    for index, item in enumerate(data):
        # Extract the user message content and sentence_id
        user_message = item['messages'][0]['content']
        sentence_id = item['metadata']['sentence_id']
        
        formatted_item = {
            "custom_id": str(sentence_id),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "temperature": temperature,
                "messages": [
                    {
                        "role": "user",
                        "content": user_message
                    }
                ]
            }
        }
        formatted_data.append(formatted_item)
    return formatted_data


def get_finetuned_model():
    finetuner = FineTuningHelper(settings)
    job = finetuner.try_load_job()
    if job and job['status'] == 'succeeded':
        fine_tuned_model = job['fine_tuned_model']
        return fine_tuned_model
    else:
        logger.info(f"Fine-tuning model is not ready yet: {job}")
        return None


if __name__ == "__main__":
    main()

