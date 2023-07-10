import argparse
import glob

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ArabicDigits")

    args = vars(parser.parse_args())

    dataset_path = f"../data/phase2/processed/{args['dataset']}/*.csv"
    output_path = f"../data/phase2/processed/{args['dataset']}/merged.csv"

    dataframes = []

    for fname in glob.glob(dataset_path):
        df = pd.read_csv(fname, encoding="cp949")
        dataframes.append(df)

    pd.concat(dataframes).to_csv(output_path)
