import csv
import pandas as pd
import os

input_path = "../data/phase2/raw/JapaneseVowels/"
output_path = "../data/phase2/processed/JapaneseVowels/"

data_file = "ae.test"
label_file = "size_ae.test"

data_file = os.path.join(input_path, data_file)
label_file = os.path.join(input_path, label_file)

output_file = "japanese_vowel_test.csv"
output_file = os.path.join(output_path, output_file)

labels = pd.read_csv(label_file, delim_whitespace=True, header=None)
block_limits = labels.iloc[0].tolist()

with open(output_file, "w", newline="") as f_out:
    writer = csv.writer(f_out)
    writer.writerow(
        [
            "a1",
            "a2",
            "a3",
            "a4",
            "a5",
            "a6",
            "a7",
            "a8",
            "a9",
            "a10",
            "a11",
            "a12",
            "block",
            "label",
        ]
    )

    with open(data_file, "r") as f_in:
        lines = f_in.readlines()
        label_index = 0
        in_block_index = 0

        for line in lines:
            line = line.strip()

            if line == "":
                in_block_index += 1

                if in_block_index == block_limits[label_index]:
                    in_block_index = 0
                    label_index += 1

            else:
                variables = line.split(" ")
                row = variables + [in_block_index, label_index]
                writer.writerow(row)
