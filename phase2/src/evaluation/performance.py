from sklearn.metrics import silhouette_score, adjusted_rand_score, rand_score
import numpy as np


def calculate_silhouette_score(data_array, labels):
    num_segments, num_length, num_dimension = data_array.shape
    data_2d = np.reshape(data_array, (num_segments, num_length * num_dimension))

    silhouette_avg = silhouette_score(data_2d, labels)

    return silhouette_avg


def calculate_adjusted_rand_index(true_labels, predicted_labels):
    ari = adjusted_rand_score(true_labels, predicted_labels)

    return ari


def calculate_rand_index(true_labels, predicted_labels):
    ri = rand_score(true_labels, predicted_labels)

    return ri


def calculate_precision(num_clusters, pred_labels, true_labels):
    counts = []

    for k in range(num_clusters):
        max_count = 0

        for j in range(num_clusters):
            count = 0

            for i in range(len(true_labels)):
                if pred_labels[i] == k:
                    if true_labels[i] == j:
                        count += 1

            max_count = max(count, max_count)

        counts.append(max_count)

    return sum(counts) / len(true_labels)


def calculate_accuracy(true_labels, predicted_labels):
    num_instances = len(true_labels)
    correct_predictions = 0

    for i in range(num_instances):
        if true_labels[i] == predicted_labels[i]:
            correct_predictions += 1

    accuracy = correct_predictions / num_instances

    return accuracy
