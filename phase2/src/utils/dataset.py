import glob

import pandas as pd
import numpy as np
from tqdm import tqdm


def load_original_dataset_envi(path):
    dataframes = []

    for file_name in glob.glob(path):
        df = pd.read_csv(file_name, encoding="cp949")
        info = file_name.split(" ")[1]
        dataframes.append((info, df))

    return dataframes


def load_event_dataset(path, skip_header=True):
    dataframes = []
    labels = []

    print("load data from each CSV file ...")
    for file_name in tqdm(glob.glob(path)):
        label = int(file_name.split("/")[-1].split(".")[0].split("_")[-1])
        labels.append(label)

        df = pd.read_csv(file_name, encoding="cp949")
        dataframes.append(df)

    print("load data from each CSV file ... Done.")

    return dataframes, labels


def load_event_dataset_envi(path, skip_header=True):
    dataframes = []
    labels = []

    for file_name in glob.glob(path):
        label = int(file_name.split("_")[1][-1]) - 1
        labels.append(label)

        df = pd.read_csv(file_name, encoding="cp949")
        dataframes.append(df)

    return dataframes, labels


def concat_dataframes(dataframes):
    modified_dataframes = []
    for i, segment in enumerate(dataframes):
        segment["segment_id"] = i
        segment["segment_id"] = segment["segment_id"].astype(object)
        modified_dataframes.append(segment)

    concatenated_data = pd.concat(modified_dataframes, ignore_index=True)

    return concatenated_data


def dataframes_to_numpy(dataframes):
    arrays_list = []
    max_timestamps = 0

    for df in dataframes:
        array = df.values
        arrays_list.append(array)
        max_timestamps = max(max_timestamps, array.shape[0])

    num_of_variables = arrays_list[0].shape[1]
    final_array = np.zeros((len(arrays_list), max_timestamps, num_of_variables))

    for i, array in enumerate(arrays_list):
        final_array[i, : array.shape[0], :] = array

    return final_array


if __name__ == "__main__":
    path = "../data/segments/*.csv"
