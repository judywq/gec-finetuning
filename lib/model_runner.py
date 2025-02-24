import os
import asyncio
import logging
from lib.finetuning_helper import FineTuningHelper
from lib.api_request_parallel_processor import process_api_requests_from_file_openai


logger = logging.getLogger(__name__)

class ModelRunner:
    def __init__(self, config, run_top_k=-1) -> None:
        self.config = config
        self.run_top_k = run_top_k
        
    def run(self, baseline=True, fine_tuned=True, skip_if_exists=True):
        if baseline:
            self._run_baseline_models(skip_if_exists=skip_if_exists)
        if fine_tuned:
            self._run_openai_finetuned(skip_if_exists=skip_if_exists)

    def _run_openai_finetuned(self, skip_if_exists=True):
        finetuner = FineTuningHelper(self.config)
        job = finetuner.try_load_job()
        if job and job['status'] == 'succeeded':
            fine_tuned_model = job['fine_tuned_model']
            input_fn = self.config.dataset_test_filename
            output_fn = self.config.dataset_test_result_finetuned_filename
            
            if skip_if_exists and os.path.exists(output_fn):
                logger.info(f"Skip running model {fine_tuned_model}.")
            else:
                self._run_openai_model(
                    input_jsonl_fn=input_fn,
                    output_jsonl_fn=output_fn,
                    model=fine_tuned_model,
                    temperature=self.config.inference_finetuned_model_temperature,
                )
        else:
            logger.info(f"Fine-tuning model is not ready yet: {job}")


    def _run_baseline_models(self, skip_if_exists=True):
        model_id = self.config.inference_base_model_id
        input_fn = self.config.dataset_test_filename
        if self.run_top_k > 0:
            input_fn = self.create_short_file(input_fn)
        output_fn = self.config.dataset_test_result_baseline_filename
        
        if skip_if_exists and os.path.exists(output_fn):
            logger.info(f"Skip running model {model_id}.")
            return
        
        self._run_openai_model(
            input_jsonl_fn=input_fn,
            output_jsonl_fn=output_fn,
            model=model_id,
            temperature=self.config.inference_base_model_temperature,
        )
    
    def create_short_file(self, input_fn):
        short_fn = input_fn.replace(".jsonl", f"_top{self.run_top_k}.jsonl")
        with open(short_fn, 'w') as f:
            for i, line in enumerate(open(input_fn)):
                f.write(line)
                if i + 1 >= self.run_top_k:
                    break
        return short_fn
        

    @classmethod
    def _run_openai_model(
        cls,
        input_jsonl_fn,
        output_jsonl_fn,
        model=None,
        temperature=0,
        max_attempts=5,
        logging_level=logging.INFO,
    ):
        logger.info(f"Run model [{model}] with input: {input_jsonl_fn}.")
        # If model and temperature are None, the value in the input file will be used.
        api_key = os.getenv("OPENAI_API_KEY_BASELINE")
        request_url = "https://api.openai.com/v1/chat/completions"
        max_requests_per_minute = 3_000 * 0.5
        max_tokens_per_minute = 250_000 * 0.5
        token_encoding_name = "cl100k_base"

        additional_params = {}
        if model is not None:
            additional_params["model"] = model
        if temperature is not None:
            additional_params["temperature"] = temperature

        # run script
        asyncio.run(
            process_api_requests_from_file_openai(
                requests_filepath=input_jsonl_fn,
                save_filepath=output_jsonl_fn,
                request_url=request_url,
                api_key=api_key,
                max_requests_per_minute=max_requests_per_minute,
                max_tokens_per_minute=max_tokens_per_minute,
                token_encoding_name=token_encoding_name,
                max_attempts=max_attempts,
                logging_level=logging_level,
                additional_params=additional_params,
            )
        )
