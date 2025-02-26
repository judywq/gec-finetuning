from lib.utils import setup_log
from lib.data_formatter import DataFormatter
import settings
import os


file_pairs = [
    ('gpt-4o_baseline', settings.dataset_test_result_gpt_4o_baseline_filename, settings.gpt_4o_baseline_results_excel),
    ('gpt-4o_finetuned', settings.dataset_test_result_gpt_4o_finetuned_filename, settings.gpt_4o_finetuned_results_excel),
    ('deepseek_baseline', settings.dataset_test_result_deepseek_baseline_filename, settings.deepseek_baseline_results_excel),
]

def main():
    formatter = DataFormatter(settings)
    formatter.run(file_pairs)


if __name__ == "__main__":
    setup_log()
    main()
