import os
import argparse
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from pprint import pprint

from phase2.clustering.clustering import *
from phase2.clustering.find_cluster import *
from phase2.feature.extract import *
from utils.dataset import *

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

    features = extract_t2f(dataframes_to_numpy(modified_dataframes), batch_size=1000)
    print(features)

    total_results = defaultdict(list)

    for iteration in range(10):
        features_dataframe = features

        k = numbers_true[args["dataset"]]  # Exact Number of k
        max_k = k + 10  # Maximum Limit of k
        min_k = k - 5  # Maximum Limit of k

        feature_X = features_dataframe.values  # Extracted Features of each segment
        feature_X[np.isnan(feature_X)] = 0  # handling NaN values as 0

        X = pd.concat(modified_dataframes).values

        X_series = dataframes_to_numpy(modified_dataframes)  # Time-series segments
        num_instances, num_timestamps, num_variables = X_series.shape

        X = dataframes_to_numpy(modified_dataframes).reshape(
            num_instances, num_timestamps * num_variables
        )

        scaler = StandardScaler()

        results = {}

        print("** Number of Clusters **")
        print("True number of Clusters of dataset=", k)

        print("Finding Number of Clusters (KMeans-sil-fe) ...")
        pred_k = find_number_of_clusters_silhouette(feature_X, min_k, max_k)
        results["k_kmeans_silhoutte_fe"] = pred_k
        total_results["k_kmeans_silhoutte_fe"].append(pred_k)

        print("Finding Number of Clusters (KMeans-gap-fe) ...")
        pred_k = find_number_of_clusters_gap(feature_X, max_k)
        results["k_kmeans_gap_fe"] = pred_k
        total_results["k_kmeans_gap_fe"].append(pred_k)

        print("Finding Number of Clusters (KMeans-info-fe) ...")
        pred_k = find_number_of_clusters_information(feature_X, min_k, max_k)
        results["k_kmeans_bic_fe"] = pred_k
        total_results["k_kmeans_bic_fe"].append(pred_k)

        print("Finding Number of Clusters (KMeans-info-fe) ...")
        pred_k = find_number_of_clusters_information(feature_X, min_k, max_k, criterion="aic")
        results["k_kmeans_aic_fe"] = pred_k
        total_results["k_kmeans_aic_fe"].append(pred_k)

        print("Finding Number of Clusters (KMeans-elbow-fe) ...")
        pred_k = find_number_of_clusters_elbow(feature_X, min_k, max_k)
        results["k_kmeans_elbow_fe"] = pred_k
        total_results["k_kmeans_elbow_fe"].append(pred_k)

        print("Finding Number of Clusters (KMeans-sil-ts) ...")
        pred_k = find_number_of_clusters_silhouette(X, min_k, max_k)
        results["k_kmeans_silhoutte_ts"] = pred_k
        total_results["k_kmeans_silhoutte_ts"].append(pred_k)

        print("Finding Number of Clusters (KMeans-elbow-ts) ...")
        pred_k = find_number_of_clusters_elbow(X, 1, max_k)
        results["k_kmeans_elbow_ts"] = pred_k
        total_results["k_kmeans_elbow_ts"].append(pred_k)

        print("Finding Number of Clusters (xmeans-ts) ...")
        pred_k = xmeans_clustering_num(X, max_k)
        results["k_xmeans_ts"] = pred_k
        total_results["k_xmeans_ts"].append(pred_k)

        if iteration == 0:
            print("Finding Number of Clusters (Agg-ts) ...")
            pred_k = agglomerative_clustering_num(X)
            results["k_agg_ts"] = pred_k
            total_results["k_agg_ts"].append(pred_k)

            print("Finding Number of Clusters (dbscan-ts) ...")
            pred_k = dbscan_num(X)
            results["k_dbscan_ts"] = pred_k
            total_results["k_dbscan_ts"].append(pred_k)

            print("Finding Number of Clusters (optics-ts) ...")
            pred_k = optics_num(X)
            results["k_optics_ts"] = pred_k
            total_results["k_optics_ts"].append(pred_k)

        print("Iteration=", iteration)
        pprint(results)
        print()

    pprint(total_results)
