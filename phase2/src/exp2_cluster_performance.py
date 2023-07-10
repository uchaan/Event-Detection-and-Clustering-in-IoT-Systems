import os
import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from pprint import pprint

from phase2.clustering.clustering import *
from phase2.clustering.find_cluster import *
from phase2.feature.extract import *
from utils.dataset import *
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

    X_series = dataframes_to_numpy(modified_dataframes)  # Time-series segments
    num_instances, num_timestamps, num_variables = X_series.shape

    X = dataframes_to_numpy(modified_dataframes).reshape(
        num_instances, num_timestamps * num_variables
    )

    features = extract_t2f(dataframes_to_numpy(modified_dataframes), batch_size=1000)
    print(features)

    features_dataframe = features

    k = numbers_true[dataset_name]  # Exact Number of k

    total_results = {}
    total_results["feature_kmeans"] = []
    total_results["kmeans_dtw"] = []
    total_results["kmeans_euc"] = []
    total_results["xmeans"] = []
    total_results["dbscan"] = []
    total_results["optics"] = []
    total_results["agg"] = []

    for iteration in range(10):
        feature_X = features_dataframe.values  # Extracted Features of each segment
        feature_X[np.isnan(feature_X)] = 0  # handling NaN values as 0

        results = {}

        scaler = StandardScaler()

        pred_labels = kmeans_clustering_vectors(feature_X, k)
        ri = round(calculate_rand_index(labels, pred_labels), 4)
        results["feature_kmeans"] = {
            "ri": ri,
        }

        total_results["feature_kmeans"].append(ri)

        pred_labels = kmeans_clustering_tslearn_dtw(X_series, k)
        ri = round(calculate_rand_index(labels, pred_labels), 4)
        results["kmeans_dtw"] = {
            "ri": ri,
        }

        total_results["kmeans_dtw"].append(ri)

        pred_labels = kmeans_clustering_tslearn_euclidean(X_series, k)
        ri = round(calculate_rand_index(labels, pred_labels), 4)
        results["kmeans_euc"] = {
            "ri": ri,
        }

        total_results["kmeans_euc"].append(ri)

        pred_labels = xmeans_clustering(X)
        ri = round(calculate_rand_index(labels, pred_labels), 4)
        results["xmeans"] = {
            "ri": ri,
        }

        total_results["xmeans"].append(ri)

        if iteration == 0:
            pred_labels = dbscan(X)
            ri = round(calculate_rand_index(labels, pred_labels), 4)
            results["dbscan"] = {
                "ri": ri,
            }

            total_results["dbscan"].append(ri)

            pred_labels = optics(X)
            ri = round(calculate_rand_index(labels, pred_labels), 4)
            results["optics"] = {
                "ri": ri,
            }

            total_results["optics"].append(ri)

            pred_labels = agglomerative_clustering(X)
            ri = round(calculate_rand_index(labels, pred_labels), 4)
            results["agg"] = {
                "ri": ri,
            }

            total_results["agg"].append(ri)

        print("iteration=", iteration)
        print()
        pprint(results)
        print()

    print(total_results)
