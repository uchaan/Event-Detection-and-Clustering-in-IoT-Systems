import os
from itertools import combinations

import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from fastdtw import fastdtw


DATA_DIR = "./data"


def prepare_dataset():
    data_segments = []
    labels = []

    for fname in os.listdir(DATA_DIR):
        space_type, day, hour, event_type, segment_len = fname.split(".")[0].split(
            "_"
        )  # '2_5_1_1_60'

        if space_type:
            file_path = os.path.join(DATA_DIR, fname)
            df = pd.read_csv(file_path)
            data_segments.append(df.values)
            labels.append(int(event_type))

    return data_segments, labels


def rand_index(true_labels, predicted_labels):
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    pairs = combinations(range(len(true_labels)), 2)

    for i, j in pairs:
        if true_labels[i] == true_labels[j] and predicted_labels[i] == predicted_labels[j]:
            true_positives += 1
        elif true_labels[i] != true_labels[j] and predicted_labels[i] != predicted_labels[j]:
            true_negatives += 1
        elif true_labels[i] == true_labels[j] and predicted_labels[i] != predicted_labels[j]:
            false_negatives += 1
        elif true_labels[i] != true_labels[j] and predicted_labels[i] == predicted_labels[j]:
            false_positives += 1

    rand_index = (true_positives + true_negatives) / (
        true_positives + false_positives + false_negatives + true_negatives
    )

    return rand_index


def precision(num_samples, num_clusters, cluster_labels, labels):
    counts = []

    for k in range(num_clusters):
        max_count = 0

        for j in range(num_clusters):
            count = 0

            for i in range(num_samples):
                if cluster_labels[i] == k:
                    if labels[i] == j:
                        count += 1

            max_count = max(count, max_count)
        counts.append(max_count)

    return sum(counts) / num_samples


def adjusted_rand_index(ground_labels, cluster_labels):
    score = adjusted_rand_score(ground_labels, cluster_labels)
    return score


def dtw_clustering(data, num_clusters):
    num_samples, sequence_length, num_variables = data.shape
    distance_matrix = np.zeros((num_samples, num_samples))

    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            distance, _ = fastdtw(data[i], data[j], dist=euclidean)
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    kmeans = KMeans(n_clusters=num_clusters)
    cluster_labels = kmeans.fit_predict(distance_matrix)

    return cluster_labels


if __name__ == "__main__":
    # datasets
    data_segments, labels = prepare_dataset()
    num_samples = len(data_segments)
    sequence_length = len(data_segments[0])
    num_variables = len(data_segments[0][0])

    for i in range(len(labels)):
        labels[i] -= 1

    print((num_samples, sequence_length, num_variables))

    # hyperparameters
    max_iteration = 100
    num_clusters = 4

    # variables
    error = 0

    # 0. convert data segments to numpy object
    data = np.concatenate(data_segments, axis=0)
    reshaped_data = np.reshape(data, (num_samples, sequence_length, num_variables))

    # 1. Normalize all data
    for i in range(num_samples):
        mean_variables = reshaped_data[i].mean(axis=0)  # [1.59 20.827 61.22 720.33]
        reshaped_data[i] = reshaped_data[i] - mean_variables

    cluster_labels = list(dtw_clustering(reshaped_data, num_clusters))

    print(adjusted_rand_index(labels, cluster_labels))
    print(precision(num_samples, num_clusters, cluster_labels, labels))
    print(rand_index(labels, cluster_labels))
