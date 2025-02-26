import json
import os
import asyncio
import litellm
from tqdm import tqdm
import argparse
import time

from lib.utils import backup_output_file

# litellm._turn_on_debug()

async def batch_process_jsonl_file(input_file, output_file, model_name, temperature=0, max_tokens=None, batch_size=10, requests_per_minute=60):
    """
    Process a JSONL file with DeepSeek model using a sliding window of concurrent tasks.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        model_name: DeepSeek model name
        temperature: Temperature for generation
        max_tokens: Maximum tokens for generation
        batch_size: Maximum number of concurrent tasks
        requests_per_minute: Maximum number of requests per minute to avoid rate limits
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(input_file, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()
    
    delay_between_requests = 60.0 / requests_per_minute if requests_per_minute > 0 else 0
    
    # Create a semaphore to limit concurrent tasks
    semaphore = asyncio.Semaphore(batch_size)
    
    async def process_with_rate_limit(line):
        async with semaphore:
            start_time = time.time()
            try:
                data = json.loads(line)
                messages = data.get("messages", [])
                metadata = data.get("metadata", {})
                result = await process_request(model_name, messages, metadata, temperature, max_tokens)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                result = [
                    {"messages": []},
                    {"error": f"JSON parse error: {str(e)}"},
                    {}
                ]
            except Exception as e:
                print(f"Error processing request: {e}")
                result = [
                    {"messages": messages if 'messages' in locals() else []},
                    {"error": str(e)},
                    metadata if 'metadata' in locals() else {}
                ]
            
            # Apply rate limiting
            elapsed = time.time() - start_time
            if elapsed < delay_between_requests:
                delay_needed = delay_between_requests - elapsed
                print(f"Rate limiting: Sleeping for {delay_needed:.2f} seconds")
                await asyncio.sleep(delay_needed)
            
            return result
    
    # Create tasks for all lines
    tasks = [process_with_rate_limit(line) for line in lines]
    
    # Process all tasks with progress bar
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for result in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing requests"):
            result_data = await result
            f_out.write(json.dumps(result_data) + '\n')
            f_out.flush()  # Ensure results are written immediately

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
    
    backup_output_file(args.output)
    
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
