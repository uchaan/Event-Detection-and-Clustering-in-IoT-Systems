import argparse
import os

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ArabicDigits")

    args = vars(parser.parse_args())

    dataset_path = f"../data/phase2/processed/{args['dataset']}/merged.csv"
    output_path = f"../data/phase2/processed/{args['dataset']}/segments/"

    df = pd.read_csv(dataset_path, encoding="cp949")

    if args["dataset"] == "LIBRAS":
        block_break = 45
        segments = []
        temp = []

        for i, row in df.iterrows():
            temp.append(list(row[:-1]))
            if row[0].astype(int) == 44:
                segments.append({"segment": temp, "label": row[-1].astype(int) - 1})
                temp = []

        for i, dic in enumerate(segments):
            segment = pd.DataFrame(dic["segment"], columns=df.columns[:-1])
            file_name = f"segment_{i}_{dic['label']}.csv"
            segment.to_csv(os.path.join(output_path, file_name))

    else:
        curr_block = 0
        curr_digit = 0
        segments = []

        temp_df = pd.DataFrame(columns=df.columns[:-2])
        temp = []

        for i, row in df.iterrows():
            if curr_block != row[-2].astype(int):
                segments.append(
                    {
                        "segment": temp,
                        "label": row[-1].astype(int),
                        "len_segment": len(temp),
                    }
                )
                temp = []
                curr_block = row[-2].astype(int)
            else:
                temp.append(list(row[:-2]))

        segments.append(
            {
                "segment": temp,
                "label": row[-1].astype(int),
                "len_segment": len(temp),
            }
        )

        for i, dic in enumerate(segments):
            segment = pd.DataFrame(dic["segment"], columns=df.columns[:-2])
            file_name = f"segment_{i}_{dic['label']}.csv"
            segment.to_csv(os.path.join(output_path, file_name))
