import argparse
import os
import warnings
import pickle

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

from phase2.clustering.clustering import *
from phase2.clustering.find_cluster import *
from phase2.feature.extract import *
from phase3.generate_rule import *
from phase3.classify import *
from utils.dataset import *
from visualization.visualize import *
from evaluation.performance import *

count = 0
sum_distance = 0
total_max = 0
total_min = 0


def find_distance(true_centroids, pred_centroids):
    global sum_distance, count, total_max, total_min
    total_distance = 0

    for true_centroid in true_centroids:
        min_distance = float("inf")

        for pred_centroid in pred_centroids:
            distance = dtw_distance(pred_centroid, true_centroid)

            total_max = max(total_max, distance)
            total_min = min(total_min, distance)

            sum_distance += distance
            count += 1

            min_distance = min(min_distance, distance)

        total_distance += min_distance

    return total_distance


numbers_true = {"ArabicDigits": 10, "CharacterTrajectories": 20, "JapaneseVowels": 9, "LIBRAS": 15}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ArabicDigits")

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

    with open(os.path.join(dataset_path, "pred_labels.pkl"), "rb") as f:
        pred_labels = pickle.load(f)

    with open(os.path.join(dataset_path, "pred_events_sampled_reduced.pkl"), "rb") as f:
        pred_events_sampled = pickle.load(f)

    with open(os.path.join(dataset_path, "pred_labels_sampled.pkl"), "rb") as f:
        pred_labels_sampled = pickle.load(f)

    with open(os.path.join(dataset_path, "pred_labels_xmeans.pkl"), "rb") as f:
        xmeans_labels = pickle.load(f)

    with open(os.path.join(dataset_path, "pred_labels_xmeans_sampled.pkl"), "rb") as f:
        xmeans_labels_sampled = pickle.load(f)

    modified_dataframes = []

    for event in actual_events:
        modified_dataframes.append(pd.DataFrame(event))

    X = dataframes_to_numpy(modified_dataframes)

    modified_dataframes = []

    for event in pred_events:
        modified_dataframes.append(pd.DataFrame(event))

    X_pred = dataframes_to_numpy(modified_dataframes)
    num_instances, num_timestamps, num_variables = X.shape

    modified_dataframes = []
    for event in pred_events_sampled:
        modified_dataframes.append(pd.DataFrame(event))

    X_pred_sampled = dataframes_to_numpy(modified_dataframes)

    max_iterations = 15

    # Actual
    labels = np.array(labels)
    actual_cluster_index_list = []

    for label in np.unique(labels):
        index = np.where(labels == label)[0]
        actual_cluster_index_list.append(index)

    actual_centroids = []

    for i, index_list in enumerate(tqdm(actual_cluster_index_list)):
        cluster = X[index_list]
        centroid = DBA(cluster, max_iterations=max_iterations)
        actual_centroids.append(centroid)

    # Predicted
    pred_labels = np.array(pred_labels)
    pred_cluster_index_list = []

    for label in np.unique(pred_labels):
        index = np.where(label == pred_labels)[0]
        pred_cluster_index_list.append(index)

    pred_centroids = []

    for i, index_list in enumerate(tqdm(pred_cluster_index_list)):
        cluster = X_pred[index_list]
        centroid = DBA(cluster, max_iterations=max_iterations)
        pred_centroids.append(centroid)

    # Predicted with sampled
    pred_labels_sampled = np.array(pred_labels_sampled)
    pred_sampled_cluster_index_list = []

    for label in np.unique(pred_labels_sampled):
        index = np.where(label == pred_labels_sampled)[0]
        pred_sampled_cluster_index_list.append(index)

    pred_sampled_centroids = []

    for i, index_list in enumerate(tqdm(pred_sampled_cluster_index_list)):
        cluster = X_pred_sampled[index_list]
        centroid = DBA(cluster, max_iterations=max_iterations)
        pred_sampled_centroids.append(centroid)

    # X-means
    xmeans_labels = np.array(xmeans_labels)
    xmeans_cluster_index_list = []

    for label in np.unique(xmeans_labels):
        index = np.where(label == xmeans_labels)[0]
        xmeans_cluster_index_list.append(index)

    xmeans_centroids = []

    for i, index_list in enumerate(tqdm(xmeans_cluster_index_list)):
        cluster = X_pred[index_list]
        centroid = DBA(cluster, max_iterations=max_iterations)
        xmeans_centroids.append(centroid)

    xmeans_labels_sampled = np.array(xmeans_labels_sampled)
    xmeans_sampled_cluster_index_list = []

    for label in np.unique(xmeans_labels_sampled):
        index = np.where(label == xmeans_labels_sampled)[0]
        xmeans_sampled_cluster_index_list.append(index)

    xmeans_sampled_centroids = []

    for i, index_list in enumerate(tqdm(xmeans_sampled_cluster_index_list)):
        cluster = X_pred_sampled[index_list]
        centroid = DBA(cluster, max_iterations=max_iterations)
        xmeans_sampled_centroids.append(centroid)

    pred = find_distance(actual_centroids, pred_centroids)
    print(pred)

    pred_sampled = find_distance(actual_centroids, pred_sampled_centroids)
    print(pred_sampled)

    xmean = find_distance(actual_centroids, xmeans_centroids)
    print(xmean)

    xmeans_sampled = find_distance(actual_centroids, xmeans_sampled_centroids)
    print(xmeans_sampled)

    print(total_max)
    print(total_min)

    print(sum_distance / count)
