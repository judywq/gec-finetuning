# It might be easier to use the web UI to create a batch job.
# https://platform.openai.com/batches/

from openai import OpenAI
import time

from lib.io import save_to_json, read_json
import settings

# Initialize OpenAI client

client = OpenAI()

batch_file_filename = "data/output/job/20250225/batch_file.json"
batch_job_filename = "data/output/job/20250225/batch_job.json"

def upload_batch_job(openai_batch_file):
    """Run the batch processing workflow."""
    # Submit batch job
    with open(openai_batch_file, 'rb') as f:
        batch_file = client.files.create(
            file=f,
            purpose="batch",
        )
    
    print(f"Batch file uploaded with ID: {batch_file.id}")
    save_to_json(batch_file.to_dict(), batch_file_filename)
    
    return batch_file.id

    
def start_batch_job(file_id):
    batch_job = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    ) 
    
    print(f"Batch job started with ID: {batch_job.id}")
    save_to_json(batch_job.to_dict(), batch_job_filename)



def main():
    openai_batch_file = settings.dataset_test_openai_batch_filename
    
    # Run batch processing
    file_id = upload_batch_job(openai_batch_file)
    start_batch_job(file_id)

if __name__ == "__main__":
    main()

