import json
import re
import pandas as pd
import logging
import os
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

class DataFormatter:
    def __init__(self, config) -> None:
        self.config = config
        
    def run(self, file_pairs: List[Tuple[str, str, str]], skip_if_exists: bool = True):
        """Main method to run the formatting process"""
        # Create output directory if it doesn't exist
        os.makedirs(self.config.excel_output_dir, exist_ok=True)
        
        for model_name, input_file_jsonl, output_file_excel in file_pairs:
            if skip_if_exists and os.path.exists(output_file_excel):
                logger.info(f"Skipping {output_file_excel} because it already exists.")
                continue
            if not os.path.exists(input_file_jsonl):
                logger.warning(f"Input file {input_file_jsonl} does not exist.")
                continue
            self.format_results(
                model_name=model_name,
                result_file=input_file_jsonl,
                output_file=output_file_excel
            )

    def format_results(self, model_name: str, result_file: str, output_file: str):
        """Format results from a jsonl file into an Excel file"""
        if not os.path.exists(result_file):
            logger.warning(f"Result file {result_file} does not exist.")
            return
            
        # Read and process the results
        results = []
        error_count = 0
        with open(result_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    # Each line contains [request, response, metadata]
                    data = json.loads(line)
                    result = self._process_result(model_name, data)
                    results.append(result)
                    if result.get(f'{model_name}_error', "") != "":
                        error_count += 1
                except Exception as e:
                    logger.error(f"Error processing line: {e}")
                    continue
        
        logger.info(f"Error count: {error_count}")
        # Convert to DataFrame and save
        if results:
            df = pd.DataFrame(results)
            df.to_excel(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
        else:
            logger.warning("No results to save")

    def _process_result(self, model_name: str, data: List) -> Dict:
        """Process a single result line"""
        request, response, metadata = data
        
        # Extract original and corrected sentences from metadata
        original = metadata.get('original', '')
        corrected = metadata.get('corrected', '')

        llm_corrected = ''        
        error_message = ''
        raw_content = ''
        # Extract LLM's correction from the response
        try:
            # The response is in the format: {"choices": [{"message": {"content": '{"corrected": "..."}'}}]}
            try:
                raw_content = response.get('choices', [{}])[0].get('message', {}).get('content', '{}')
            except (KeyError, IndexError, AttributeError) as e:
                logger.error(f"Error extracting raw content: {e}")
                raw_content = ''
            content = self._extract_json_markdown(raw_content)
            content = self._extract_json_content(content)
            try:
                json_content = json.loads(content)
            except json.JSONDecodeError as e:
                content = self.escape_quotes_in_json_values(content)
                json_content = json.loads(content)
            llm_corrected = json_content.get('corrected', '')
        except Exception as e:
            logger.error(f"Error extracting {model_name} correction: {e}")
            error_message = str(e)
        
        return {
            'original': original,
            'corrected': corrected,
            f'{model_name}_corrected': llm_corrected,
            f'{model_name}_response': response,
            f'{model_name}_raw_content': raw_content,
            f'{model_name}_cleaned_content': content,
            f'{model_name}_error': error_message
        }
    
    def _extract_json_markdown(self, content: str) -> str:
        """Extract the JSON content in markdown format from the response"""
        json_pattern = r'```json\s*({[\s\S]*?})\s*```'
        match = re.search(json_pattern, content)
        if match:
            return match.group(1)
        else:
            return content

    def _extract_json_content(self, content: str) -> str:
        """Extract the JSON from plain text"""
        json_pattern = r'({[\s\S]*?})\s+'
        match = re.search(json_pattern, content)
        if match:
            return match.group(1)
        else:
            return content

    @staticmethod
    def escape_quotes_in_json_values(json_string):
        """Escape any unescaped double quotes in the content"""
        pattern = r'(: *")(.*?)("\s*(?=}))'
        
        def replace_func(match):
            prefix, content, suffix = match.groups()
            # Escape any unescaped double quotes in the content
            escaped_content = content.replace('"', '\\"')
            return prefix + escaped_content + suffix
        
        return re.sub(pattern, replace_func, json_string)
    

if __name__ == "__main__":
    content = '{"corrected": "I " rappel " , "paintball" ..."}'
    # content = '{"corrected": "Ia ..."}'
    content = """{\n"corrected": "Now lookee here , " he said , " the question being whether you're to be let to live . You know"\n}"""
    print(DataFormatter.escape_quotes_in_json_values(content))

