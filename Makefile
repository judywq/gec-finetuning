ds:
	pipenv run python C01_deepseek_baseline.py \
		--input data/output/dataset/test.jsonl \
		--output data/output/result/test_result_deepseek_baseline.jsonl \
		--model deepseek/deepseek-chat \
		--batch_size 20 \
		--requests_per_minute 200 \
		--max_retries 0

openai-finetuned:
	pipenv run python C01_deepseek_baseline.py \
		--input data/output/dataset/test.jsonl \
		--output data/output/result/test_result_gpt_4o_finetuned.jsonl \
		--model ft:gpt-4o-2024-08-06:nlp-projects:grammar-correction:B4kEog4Y \
		--batch_size 3 \
		--requests_per_minute 30 \
		--max_retries 10

