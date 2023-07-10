import os
import pandas as pd
import numpy as np

input_path = "../data/phase2/raw/LIBRAS"
input_file = "movement_libras.data"

output_path = "../data/phase2/processed/LIBRAS"
output_file = "libras.csv"

input_file = os.path.join(input_path, input_file)
output_file = os.path.join(output_path, output_file)

with open(input_file, "rb") as f:
    df = pd.read_csv(f, delimiter=",", header=None)
    Xy = {}
    dfs = []

    for i, row in df.iterrows():
        new = np.reshape(df.iloc[i, :-1].values, (-1, 2))
        new = pd.DataFrame(new, columns=["att1", "att2"])
        label = df.iloc[i, -1]
        new["label"] = label
        dfs.append(new)

    pd.concat(dfs).to_csv(output_file)
