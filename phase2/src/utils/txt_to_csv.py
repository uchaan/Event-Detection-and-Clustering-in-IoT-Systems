import os
import csv


input_dir = "./dataset/data_txt/"
output_dir = "./dataset/data_csv/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(input_dir):
    print(filename)

    if filename.endswith(".txt"):
        with open(os.path.join(input_dir, filename), "r", encoding="cp949") as input_file, open(
            os.path.join(output_dir, filename[:-4] + ".csv"), "w", newline="", encoding="cp949"
        ) as output_file:
            writer = csv.writer(output_file)

            for line in input_file:
                columns = line.strip().split(",")
                writer.writerow(columns)
