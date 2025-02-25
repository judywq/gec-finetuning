import json
import os
import asyncio
import litellm
from tqdm import tqdm
import argparse
import time


async def batch_process_jsonl_file(input_file, output_file, model_name, temperature=0, max_tokens=None, batch_size=10, requests_per_minute=60):
    """
    Process a JSONL file in batches with DeepSeek model using async and save results to a new JSONL file.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        model_name: DeepSeek model name
        temperature: Temperature for generation
        max_tokens: Maximum tokens for generation
        batch_size: Number of requests to process in parallel
        requests_per_minute: Maximum number of requests per minute to avoid rate limits
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Read all lines from input file
    with open(input_file, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()
    
    # Calculate delay between requests to respect rate limits
    delay_between_requests = 60.0 / requests_per_minute if requests_per_minute > 0 else 0
    
    # Open output file for writing
    with open(output_file, 'w', encoding='utf-8') as f_out:
        # Process in batches
        for i in tqdm(range(0, len(lines), batch_size), desc="Processing batches"):
            batch_lines = lines[i:i+batch_size]
            tasks = []
            
            # Create tasks for each line in the batch
            for line in batch_lines:
                try:
                    data = json.loads(line)
                    messages = data.get("messages", [])
                    metadata = data.get("metadata", {})
                    
                    # Create a task for each request
                    task = asyncio.create_task(process_request(model_name, messages, metadata, temperature, max_tokens))
                    tasks.append(task)
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON: {e}")
                    # Write error to output
                    error_output = [
                        {"messages": []},
                        {"error": f"JSON parse error: {str(e)}"},
                        {}
                    ]
                    f_out.write(json.dumps(error_output) + '\n')
            
            # Start time for rate limiting
            batch_start_time = time.time()
            
            # Wait for all tasks in the batch to complete
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Write results to output file
            for result in batch_results:
                if isinstance(result, Exception):
                    # Handle exceptions
                    error_output = [
                        {"messages": []},
                        {"error": f"Processing error: {str(result)}"},
                        {}
                    ]
                    f_out.write(json.dumps(error_output) + '\n')
                else:
                    f_out.write(json.dumps(result) + '\n')
            
            # Calculate time spent on this batch and delay if needed to respect rate limits
            batch_duration = time.time() - batch_start_time
            expected_duration = len(batch_lines) * delay_between_requests
            
            if batch_duration < expected_duration:
                delay_needed = expected_duration - batch_duration
                print(f"Rate limiting: Sleeping for {delay_needed:.2f} seconds")
                await asyncio.sleep(delay_needed)

async def process_request(model_name, messages, metadata, temperature, max_tokens):
    """Process a single request and return the formatted result."""
    try:
        # Call the model asynchronously
        response = await litellm.acompletion(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Create the output format: [request, response, metadata]
        return [
            {"messages": messages},  # Original request
            response.to_dict(),      # Model response
            metadata                 # Original metadata
        ]
    except Exception as e:
        print(f"Error in request: {e}")
        # Return error information
        return [
            {"messages": messages},
            {"error": str(e)},
            metadata
        ]

def main():
    parser = argparse.ArgumentParser(description="Process a JSONL file with DeepSeek model")
    parser.add_argument("--input", type=str, default="data/output/dataset/test_top5.jsonl",
                        help="Input JSONL file path")
    parser.add_argument("--output", type=str, default="data/output/result/test_result_deepseek.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--model", type=str, default="deepseek-chat",
                        help="DeepSeek model name")
    parser.add_argument("--temperature", type=float, default=0,
                        help="Temperature for generation")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Maximum tokens for generation")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Number of requests to process in parallel")
    parser.add_argument("--requests_per_minute", type=int, default=60,
                        help="Maximum number of requests per minute (rate limit)")
    
    args = parser.parse_args()
    
    print(f"Processing {args.input} with model {args.model}...")
    asyncio.run(batch_process_jsonl_file(
        args.input, 
        args.output,
        args.model,
        args.temperature,
        args.max_tokens,
        args.batch_size,
        args.requests_per_minute
    ))
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
