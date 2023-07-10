import os
import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from tqdm import tqdm

from phase2.clustering.clustering import *
from phase2.clustering.find_cluster import *
from phase2.feature.extract import *
from phase3.generate_rule import *
from phase3.classify import *
from utils.dataset import *
from visualization.visualize import *
from evaluation.performance import *

numbers_true = {"ArabicDigits": 10, "CharacterTrajectories": 20, "JapaneseVowels": 9, "LIBRAS": 15}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ArabicDigits")

    args = vars(parser.parse_args())

    dataset_name = args["dataset"]
    dataset_path = f"../data/phase2/processed/{dataset_name}/segments/*.csv"

    dataframes, labels = load_event_dataset(dataset_path)

    modified_dataframes = []

    for df in tqdm(dataframes):
        if dataset_name == "CharacterTrajectories":
            modified_df = df.drop(columns=df.columns[:1])
        else:
            modified_df = df.drop(columns=df.columns[:2])
        modified_dataframes.append(modified_df)

    X = dataframes_to_numpy(modified_dataframes)
    num_instances, num_timestamps, num_variables = X.shape

    labels = np.array(labels)
    cluster_index_list = []

    for label in np.unique(labels):
        index = np.where(labels == label)[0]
        cluster_index_list.append(index)

    max_iterations = 15

    for iteration in range(1):
        print("iteration=", iteration)
        centroids = []

        for i, index_list in enumerate(tqdm(cluster_index_list)):
            cluster = X[index_list]
            centroid = DBA(cluster, max_iterations=max_iterations)
            centroids.append(centroid)

        classifications = classify_dba(X, centroids)

        accuracy = calculate_accuracy(labels, classifications)
        ri = calculate_rand_index(labels, classifications)
        ari = calculate_adjusted_rand_index(labels, classifications)

        print("accuracy=", accuracy)
        print("randindex=", ri)
        print("adjusted randindex=", ari)
