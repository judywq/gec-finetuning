import json
import pandas as pd
import logging
import os
from typing import List, Dict

logger = logging.getLogger(__name__)

class DataFormatter:
    def __init__(self, config) -> None:
        self.config = config
        
    def run(self):
        """Main method to run the formatting process"""
        # Create output directory if it doesn't exist
        os.makedirs(self.config.excel_output_dir, exist_ok=True)
        
        # Format baseline results
        self.format_results(
            result_file=self.config.dataset_test_result_baseline_filename,
            output_file=self.config.baseline_results_excel
        )
        
        # Format finetuned results if they exist
        if os.path.exists(self.config.dataset_test_result_finetuned_filename):
            self.format_results(
                result_file=self.config.dataset_test_result_finetuned_filename,
                output_file=self.config.finetuned_results_excel
            )

    def format_results(self, result_file: str, output_file: str):
        """Format results from a jsonl file into an Excel file"""
        if not os.path.exists(result_file):
            logger.warning(f"Result file {result_file} does not exist.")
            return
            
        # Read and process the results
        results = []
        with open(result_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    # Each line contains [request, response, metadata]
                    data = json.loads(line)
                    result = self._process_result(data)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing line: {e}")
                    continue
        
        # Convert to DataFrame and save
        if results:
            df = pd.DataFrame(results)
            df.to_excel(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
        else:
            logger.warning("No results to save")

    def _process_result(self, data: List) -> Dict:
        """Process a single result line"""
        request, response, metadata = data
        
        # Extract original and corrected sentences from metadata
        original = metadata.get('original', '')
        corrected = metadata.get('corrected', '')
        
        # Extract OpenAI's correction from the response
        try:
            # The response is in the format: {"choices": [{"message": {"content": '{"corrected": "..."}'}}]}
            raw_content = response.get('choices', [{}])[0].get('message', {}).get('content', '{}')
            content = self._preprocess_content(raw_content)
            openai_response = json.loads(content)
            openai_corrected = openai_response.get('corrected', '')
        except Exception as e:
            logger.error(f"Error extracting OpenAI correction: {e}")
            openai_corrected = '<ERROR>'
        
        return {
            'original': original,
            'corrected': corrected,
            'openai_corrected': openai_corrected,
            'openai_raw_response': raw_content
        }
    
    def _preprocess_content(self, content: str) -> str:
        """Preprocess the content to remove the ```json and ``` tags"""
        return content.replace('```json', '').replace('```', '')
