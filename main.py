from pathlib import Path
import pandas as pd
from errant.converter import convert_m2_file


input_files = [
    "data/m2/ABC.train.gold.bea19.m2",
    "data/m2/ABCN.dev.gold.bea19.m2",
]


def main():
    for input_file in input_files:
        ip = Path(input_file)
        output_path = ip.parent.parent / "output" / "txt"
        output_path.mkdir(parents=True, exist_ok=True)
        output_file_orig = output_path / ip.name.replace(".m2", ".orig.txt")
        output_file_cor = output_path / ip.name.replace(".m2", ".cor.txt")
        result = convert_m2_file(input_file)
        with open(output_file_orig, "w", encoding="utf-8") as f:
            for item in result:
                f.write(item["original"] + "\n")
        with open(output_file_cor, "w", encoding="utf-8") as f:
            for item in result:
                f.write(item["corrected"] + "\n")


if __name__ == "__main__":
    main()
