import os
import json
import random
from typing import Dict, List
import logging
from lib.io import save_to_jsonl
import settings

logger = logging.getLogger(__name__)

class DatasetPreparation:
    def __init__(self, config) -> None:
        self.config = config
        
    def run(self, skip_if_exist=True):
        # Training data
        self.prepare_train_val(
            original_file=self.config.train_files["original"],
            corrected_file=self.config.train_files["corrected"],
            train_output_file=self.config.dataset_train_filename,
            val_output_file=self.config.dataset_val_filename,
            skip_if_exist=skip_if_exist
        )
        
        # Validation data
        # self.prepare_train_val(
        #     original_file=self.config.val_files["original"],
        #     corrected_file=self.config.val_files["corrected"],
        #     output_file=self.config.dataset_val_filename,
        #     skip_if_exist=skip_if_exist
        # )

        # Test data
        self.prepare_test(
            original_file=self.config.test_files["original"],
            corrected_file=self.config.test_files["corrected"],
            output_file=self.config.dataset_test_filename,
            skip_if_exist=skip_if_exist 
        )

    def prepare_train_val(self, original_file: str, corrected_file: str, train_output_file: str, val_output_file: str, skip_if_exist=True):
        if skip_if_exist and os.path.exists(train_output_file):
            logger.debug(f"Dataset {train_output_file} already exists, skip.")
            return

        if skip_if_exist and os.path.exists(val_output_file):
            logger.debug(f"Dataset {val_output_file} already exists, skip.")
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
            record = self.create_chat_example(orig.strip(), corr.strip(), for_training=True)
            dataset.append(record)
        
        # Split the dataset into train and val with random shuffle
        random.shuffle(dataset)
        train_size = int(len(dataset) * self.config.train_rate)
        train_dataset = dataset[:train_size]
        val_dataset = dataset[train_size:]
            
        save_to_jsonl(train_dataset, train_output_file)
        save_to_jsonl(val_dataset, val_output_file)
        logger.info(f"Created train dataset with {len(train_dataset)} examples in {train_output_file}")
        logger.info(f"Created val dataset with {len(val_dataset)} examples in {val_output_file}")

    def prepare_test(self, original_file: str, corrected_file: str, output_file: str, skip_if_exist=True):
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

        dataset = []
        sentence_id = 1
        for orig, corr in zip(orig_lines, corr_lines):
            record = self.create_chat_example(
                original=orig.strip(), 
                corrected=corr.strip(), 
                for_training=False, 
                sentence_id=sentence_id
            )
            dataset.append(record)
            sentence_id += 1
            
        save_to_jsonl(dataset, output_file)
        logger.info(f"Created dataset with {len(dataset)} examples in {output_file}")

    def create_chat_example(self, original: str, corrected: str|None, for_training: bool=True, sentence_id: int|None=None) -> Dict:
        messages = [
            {
                "role": "user",
                "content": settings.inference_prompt_template.format(original=original)
            },
        ]
        
        if for_training:
            if corrected is not None:
                messages.append({
                    "role": "assistant",
                    "content": f'{{"corrected": "{corrected}"}}'
                })
            return {"messages": messages}
        

        return {
            "messages": messages, 
            "metadata": {
                "sentence_id": sentence_id,
                "original": original,
                "corrected": corrected if corrected is not None else ""
            }
        } 