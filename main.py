from pathlib import Path
import pandas as pd
from errant.converter import convert_m2_file


input_files = [
    "data/m2/ABC.train.gold.bea19.m2",
    "data/m2/ABCN.dev.gold.bea19.m2",
]

# input_files = [
#     "data/output/txt/ABC.train.gold.bea19.errant2_0_0.m2",
#     "data/output/txt/ABC.train.gold.bea19.errant3_0_0.m2",
# ]

def main():
    orig_suffix = ".orig.txt"
    cor_suffix = ".cor.txt"
    convert_m2_to_txt(orig_suffix=orig_suffix, cor_suffix=cor_suffix)


def convert_m2_to_txt(orig_suffix=".orig.txt", cor_suffix=".cor.txt"):
    for input_file in input_files:
        ip = Path(input_file)
        output_path = ip.parent.parent / "output" / "txt"
        output_path.mkdir(parents=True, exist_ok=True)
        result = convert_m2_file(input_file)

        if orig_suffix:
            output_file_orig = output_path / ip.name.replace(".m2", orig_suffix)
            with open(output_file_orig, "w", encoding="utf-8") as f:
                for item in result:
                    f.write(item["original"] + "\n")
            print(f"Saved {output_file_orig}")
            
        if cor_suffix:
            output_file_cor = output_path / ip.name.replace(".m2", cor_suffix)
            with open(output_file_cor, "w", encoding="utf-8") as f:
                for item in result:
                    f.write(item["corrected"] + "\n")
            print(f"Saved {output_file_cor}")


if __name__ == "__main__":
    main()
