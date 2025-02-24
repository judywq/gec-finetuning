import os
import json
from typing import Dict, List
import logging
from lib.io import save_to_jsonl

logger = logging.getLogger(__name__)

class DatasetPreparation:
    def __init__(self, config) -> None:
        self.config = config
        
    def run(self, skip_if_exist=True):
        # Training data
        self.prepare(
            original_file=self.config.train_files["original"],
            corrected_file=self.config.train_files["corrected"],
            output_file=self.config.dataset_train_filename,
            skip_if_exist=skip_if_exist
        )
        
        # Validation data
        self.prepare(
            original_file=self.config.val_files["original"],
            corrected_file=self.config.val_files["corrected"],
            output_file=self.config.dataset_val_filename,
            skip_if_exist=skip_if_exist
        )

    def prepare(self, original_file: str, corrected_file: str, output_file: str, skip_if_exist=True):
        if skip_if_exist and os.path.exists(output_file):
            logger.debug(f"Dataset {output_file} already exists, skip.")
            return
            
        if not os.path.exists(original_file):
            logger.warning(f"Original file {original_file} does not exist.")
            return
        if not os.path.exists(corrected_file):
            logger.warning(f"Corrected file {corrected_file} does not exist.")
            return
            
        with open(original_file, 'r', encoding='utf-8') as f_orig, \
             open(corrected_file, 'r', encoding='utf-8') as f_corr:
            orig_lines = f_orig.readlines()
            corr_lines = f_corr.readlines()
            
        if len(orig_lines) != len(corr_lines):
            raise ValueError("Original and corrected files have different number of lines")
            
        dataset = []
        for orig, corr in zip(orig_lines, corr_lines):
            record = self.create_chat_example(orig.strip(), corr.strip())
            dataset.append(record)
            
        save_to_jsonl(dataset, output_file)
        logger.info(f"Created dataset with {len(dataset)} examples in {output_file}")

    def create_chat_example(self, original: str, corrected: str) -> Dict:
        messages = [
            {
                "role": "user",
                "content": f"""You are an English linguist and your task is to correct the grammatical and mechanical errors in English sentences. 
Please make only necessary corrections to the extent that a sentence will be free from errors and comprehensible. 
Do not alter word choices unnecessarily (e.g., replacing words with synonyms) or make stylistic improvements. 
Also, the sentences are tokenized, which means punctuation marks are separated from the English words by spaces. 
When returning the corrected sentences, please use the same tokenized format. 
Please respond in the following JSON format:
{{
  "corrected": "..."
}}

The original sentence is:
{original}"""
            },
            {
                "role": "assistant",
                "content": f'{{"corrected": "{corrected}"}}'
            }
        ]
        
        return {"messages": messages} 