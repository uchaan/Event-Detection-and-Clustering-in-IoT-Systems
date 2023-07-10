import os
import argparse
import warnings
from collections import defaultdict
import pickle

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pprint import pprint

from phase2.clustering.clustering import *
from phase2.clustering.find_cluster import *
from phase2.feature.extract import *
from utils.dataset import *

numbers_true = {
    "ArabicDigits": 10,
    "CharacterTrajectories": 20,
    "JapaneseVowels": 9,
    "LIBRAS": 15,
    "integrated": 3,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="integrated")

    args = vars(parser.parse_args())

    dataset_name = args["dataset"]
    dataset_path = f"../data/integrated/"

    actual_events = []
    labels = []

    pred_events = []
    pred_events_sampled = []

    with open(os.path.join(dataset_path, "actual_events.pkl"), "rb") as f:
        actual_events = pickle.load(f)

    with open(os.path.join(dataset_path, "actual_events_label.pkl"), "rb") as f:
        labels = pickle.load(f)

    with open(os.path.join(dataset_path, "pred_events.pkl"), "rb") as f:
        pred_events = pickle.load(f)

    with open(os.path.join(dataset_path, "pred_events_sampled_reduced.pkl"), "rb") as f:
        pred_events_sampled = pickle.load(f)

    modified_dataframes = []

    for pred_event in pred_events_sampled:
        modified_dataframes.append(pd.DataFrame(pred_event))

    features = extract_t2f(dataframes_to_numpy(modified_dataframes), batch_size=1000)
    print(features)

    total_results = defaultdict(list)

    for iteration in range(10):
        features_dataframe = features

        k = numbers_true[args["dataset"]]  # Exact Number of k
        max_k = k + 10  # Maximum Limit of k
        min_k = 1  # Maximum Limit of k

        feature_X = features_dataframe.values  # Extracted Features of each segment
        feature_X[np.isnan(feature_X)] = 0  # handling NaN values as 0

        scaler = StandardScaler()

        results = {}

        print("** Number of Clusters **")
        print("True number of Clusters of dataset=", k)

        print("Finding Number of Clusters (KMeans-elbow-fe) ...")
        pred_k = find_number_of_clusters_elbow(feature_X, min_k, max_k)
        results["k_kmeans_elbow_fe"] = pred_k
        total_results["k_kmeans_elbow_fe"].append(pred_k)

        print("Iteration=", iteration)
        pprint(results)
        print()

    pprint(total_results)
