import os
import csv

input_path = "../data/phase2/raw/ArabicDigits"
input_files = ["Train_Arabic_Digit.txt"]

output_path = "../data/phase2/processed/ArabicDigits"
output_file = "Train_Arabic_Digit.csv"

# Define the labels for each digit
digit_labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

with open(os.path.join(output_path, output_file), "w", newline="") as f_out:
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
            "a13",
            "block",
            "digit",
        ]
    )

    for file_index, input_file in enumerate(input_files):
        input_file = os.path.join(input_path, input_file)

        with open(input_file, "r") as f_in:
            lines = f_in.readlines()

            digit_index = 0
            in_block_index = 0

            for line in lines:
                line = line.strip()

                if line == "":
                    in_block_index += 1

                    if in_block_index == 660:
                        in_block_index = 0
                        digit_index += 1

                elif line[0].isdigit():
                    variables = line.split(" ")
                    row = variables + [in_block_index, digit_index]
                    writer.writerow(row)

print(f"Processing completed. Data merged and saved to {output_file}.")
