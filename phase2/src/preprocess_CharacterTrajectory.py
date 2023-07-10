import os
import sys

import pandas as pd
import scipy.io
import numpy as np


mat_file_path = "../data/phase2/raw/CharacterTrajectories/mixoutALL_shifted.mat"

# Load the .mat file
ct_raw_data = scipy.io.loadmat(mat_file_path)

chars = ct_raw_data["consts"][0][0][3][0]

cid_map = {c[0]: i for i, c in enumerate(chars)}
rcid_map = {i: c[0] for i, c in enumerate(chars)}

ct_labels = ct_raw_data["consts"][0][0][4][0] - 1

print("Dataset contains letters: [{}]".format("".join([c[0] for c in chars])))
print("Dataset contains {} datapoints.".format(len(ct_labels)))

count = {rcid_map[i]: 0 for i in range(20)}

for l in ct_labels:
    count[rcid_map[l]] += 1

print("Individual character counts are: {}".format(count))

ct_data = ct_raw_data["mixout"][0]
ct_data = [d.T for d in ct_data]
ct_data = [d[~np.all(d == 0, axis=1), :] for d in ct_data]

print(len(ct_data))

output_path = f"../data/phase2/processed/CharacterTrajectories/segments/"

for i, series in enumerate(ct_data):
    df = pd.DataFrame(series, columns=["a1", "a2", "a3"])
    label = ct_labels[i]
    file_name = f"segment_{i}_{label}.csv"
    df.to_csv(os.path.join(output_path, file_name))
