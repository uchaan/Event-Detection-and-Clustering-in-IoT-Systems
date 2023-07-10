"""
    [Phase 2] Labeling Unlabeled Events
        - (05.29) assume that segments are given by phase 1
"""

import sys
import argparse

import numpy as np
import pandas as pd

import phase2.feature.extract as fe
import phase2.feature.normalize as scaler
import phase2.clustering.kmeans as kmeans
import phase2.clustering.agglomerative as agg
import phase2.clustering.find_cluster as ncluster
import evaluation.performance as performance
import utils.dataset as dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ArabicDigits")

    args = vars(parser.parse_args())

    # Data Loading (ENVI-met dataset)
    dataset_path = "../data/envi-met/segments/*.csv"
    dataframes, true_labels = dataset.load_event_dataset_envi(dataset_path)

    # Drop unnecessary columns
    modified_dataframes = []  # List of segments (dataframe)
    index_list = list(range(0, 60))
    for df in dataframes:
        modified_df = df.drop(
            columns=["Date", "Time", "Model time (min)"]
        )  # remove "Date", "Time", "Model Time" column
        modified_df["index"] = index_list
        modified_df.set_index("index")

        modified_dataframes.append(modified_df)

    features_list = []  # List of extracted features
    for df in modified_dataframes:
        features = fe.extract_statistic(df)
        autocorrelation = fe.extract_autocorrelation(df)
        features_list.append(features + autocorrelation)

    features_dataframe = pd.DataFrame(features_list)  # Convert to dataframe

    max_k = 10  # Maximum Limit of k
    k = 7  # Exact Number of k

    # Dataset as dataframe
    X = dataset.dataframes_to_numpy(modified_dataframes)  # Time-series segments
    feature_X = features_dataframe.values  # Extracted Features of each segment
    feature_X[np.isnan(feature_X)] = 0  # handling NaN values as 0

    num_segments, num_timestamps, num_dimension = X.shape  # for 2D shape conversion

    k_fe = 0  # Feature Extracion with Kmeans
    k_ts = 0  # Time Series with Kmeans

    # Finding Number of Clusters
    k_ts = ncluster.find_optimal_clusters(
        np.reshape(X, (num_segments * num_timestamps, num_dimension)), max_k
    )
    k_fe = ncluster.find_optimal_clusters(feature_X, max_k)

    print("** Number of Clusters **")
    print("True number of Clusters of dataset=")
    print("kmeans_feature_extraction=", k_ts)

    print("kmeans_time_series=", k_ts)
    print("k_fe")
    print(k_fe)
