import json
import os
import asyncio
import litellm
from tqdm import tqdm
import argparse
import time
import io
import logging

from lib.utils import backup_output_file, setup_log

logger = logging.getLogger(__name__)

# litellm._turn_on_debug()

async def batch_process_jsonl_file(
    input_file, 
    output_file, 
    model_name, 
    temperature=0, 
    max_tokens=None, 
    batch_size=10, 
    requests_per_minute=60, 
    dry_run=False):
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
    # Create output file directory if not dry run
    if not dry_run:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(input_file, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()
        
    # sentence_ids = list(map(lambda x: json.loads(x)['metadata']['sentence_id'], lines))
    # print(sentence_ids)
    
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
                result = await process_request(model_name, messages, metadata, temperature, max_tokens, dry_run)
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
                logger.debug(f"Rate limiting: Sleeping for {delay_needed:.2f} seconds")
                await asyncio.sleep(delay_needed)
            
            return result
    
    # Create tasks for all lines
    tasks = [process_with_rate_limit(line) for line in lines]
    
    # Process all tasks with progress bar
    output_buffer = io.StringIO() if dry_run else open(output_file, 'w', encoding='utf-8')
    try:
        for result in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing requests"):
            result_data = await result
            output_buffer.write(json.dumps(result_data) + '\n')
            if not dry_run:
                output_buffer.flush()  # Ensure results are written immediately
    finally:
        if not dry_run:
            output_buffer.close()
        
    if dry_run:
        return output_buffer.getvalue()

async def process_request(model_name, messages, metadata, temperature, max_tokens, dry_run):
    """Process a single request and return the formatted result."""
    if dry_run:
        sentence_id = metadata["sentence_id"]
        logger.debug("Processing sentence: %s", sentence_id)
        time_to_sleep = sentence_id % 5
        await asyncio.sleep(time_to_sleep)
        return [
            {"messages": messages},
            {"error": "Dry run"},
            metadata
        ]
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
    parser.add_argument("--dry-run", type=bool, default=False,
                        help="Run without sending requests or saving results")
    
    args = parser.parse_args()
    
    if not args.dry_run:
        backup_output_file(args.output)
    else:
        logger.info("-" * 20)
        logger.info("- Dry run mode -")
        logger.info("-" * 20)
    
    logger.info("Processing %s with model %s...", args.input, args.model)
    result = asyncio.run(batch_process_jsonl_file(
        args.input, 
        args.output,
        args.model,
        args.temperature,
        args.max_tokens,
        args.batch_size,
        args.requests_per_minute,
        args.dry_run
    ))
    
    if args.dry_run:
        print("Dry run output:")
        print(result)
        print("-" * 20)
    else:
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    setup_log(logging.INFO)
    main()
