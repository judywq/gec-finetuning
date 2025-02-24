from lib.utils import setup_log
from lib.model_runner import ModelRunner
import settings
import os

skip_if_exist = True
skip_if_exist = False
# run_top_k = -1
run_top_k = 5

def main():
    os.makedirs("data/output/result", exist_ok=True)
    
    runner = ModelRunner(settings, run_top_k=run_top_k)
    runner.run(
        baseline=True,
        fine_tuned=False,
        skip_if_exists=skip_if_exist
    )


if __name__ == "__main__":
    setup_log()
    main()
