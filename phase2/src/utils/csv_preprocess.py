import csv
import os


input_dir = "./data/data_csv_raw/"
output_dir = "./data/data_csv"

columns_to_save = [1, 2, 3, 8, 10, 13, 25]

value_to_match = "7.00"

for filename in os.listdir(input_dir):
    if filename.endswith(".csv"):
        with open(os.path.join(input_dir, filename), "r", encoding="cp949") as csv_file:
            reader = csv.reader(csv_file)
            header = next(reader)

            modified_header = [val for i, val in enumerate(header) if i in columns_to_save]
            modified_rows = [modified_header]

            for row in reader:
                if row[4] == value_to_match:
                    modified_row = [val for i, val in enumerate(row) if i in columns_to_save]
                    modified_rows.append(modified_row)

            with open(
                os.path.join(output_dir, "modified_" + filename), "w", newline="", encoding="cp949"
            ) as modified_csv_file:
                writer = csv.writer(modified_csv_file)
                writer.writerows(modified_rows)
