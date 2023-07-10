import os
import csv


def extract_csv(input_file, output_file, variables):
    with open(input_file, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        save_columns = [header.index(col) for col in variables]
        rows_to_write = []

        for row in reader:
            filtered_row = [value for idx, value in enumerate(row) if idx in save_columns]
            rows_to_write.append(filtered_row)

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(variables)
        writer.writerows(rows_to_write)


output_dir = "./data"
dir1 = "./dataset"
variables = [
    "wSpeed (m/s)",
    " Potential Air Temperature (°C)",
    "Relative Humidity (%)",
    "Local CO2 (mg/m³)",
]

for space in os.listdir(dir1):
    dir2 = os.path.join(dir1, space)

    for t in os.listdir(dir2):
        dir3 = os.path.join(dir2, t)

        for day in os.listdir(dir3):
            dir4 = os.path.join(dir3, day)

            for filename in os.listdir(dir4):
                filepath = os.path.join(dir4, filename)
                output_path = os.path.join(output_dir, filename)
                print(filepath)
                extract_csv(filepath, output_path, variables)
