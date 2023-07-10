from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np
from fastdtw import fastdtw
from sklearn import svm


def classify_dba(X, centroids, similarity="dtw"):
    classifications = []

    for i, series in enumerate(X):
        min_distance = float("inf")
        classification = -1

        for label, centroid in enumerate(centroids):
            if similarity == "dtw":
                distance, _ = fastdtw(series, centroid, dist=euclidean)
            else:
                distance = euclidean(series, centroid)

            if distance < min_distance:
                min_distance = distance
                classification = label

        classifications.append(classification)

    return classifications


def classify_feature(X, features):
    classifications = []

    for i, x in enumerate(X):
        min_distance = float("inf")
        classification = -1

        for label, feature in enumerate(features):
            distance = euclidean(x, feature)

            if distance < min_distance:
                min_distance = distance
                classification = label

        classifications.append(classification)

    return classifications


def classify_mixed(X, feature_X, centroids, features, alpha=0.5):
    classifications = []

    for i, x in enumerate(X):
        min_distance = float("inf")
        classification = -1

        max_distance_dba = 0
        max_distance_feature = 0

        min_distance_dba = float("inf")
        min_distance_feature = float("inf")

        series = X[i]
        feature = feature_X[i]

        for label in range(len(centroids)):
            distance_dba, _ = fastdtw(series, centroids[label], dist=euclidean)
            max_distance_dba = max(distance_dba, max_distance_dba)  # for normalization
            min_distance_dba = min(distance_dba, min_distance_dba)

            distance_feature = euclidean(feature, features[label])
            max_distance_feature = max(distance_feature, max_distance_feature)
            min_distance_feature = min(distance_feature, min_distance_feature)

        for label in range(len(centroids)):
            distance_dba, _ = fastdtw(series, centroids[label], dist=euclidean)
            distance_feature = euclidean(feature, features[label])
            distance_dba_normalized = (distance_dba - min_distance_dba) / (
                max_distance_dba - min_distance_dba
            )
            distance_feature_normalized = (distance_feature - min_distance_feature) / (
                max_distance_feature - min_distance_feature
            )
            distance_normalized = (
                alpha * distance_dba_normalized + (1 - alpha) * distance_feature_normalized
            )

            if distance_normalized < min_distance:
                min_distance = distance_normalized
                classification = label

        classifications.append(classification)

    return classifications


def classify_mixed_2(X, feature_X, centroids, features, alpha=0.5):
    classifications = []

    max_distance_dba = 0
    max_distance_feature = 0

    min_distance_dba = float("inf")
    min_distance_feature = float("inf")

    for i, x in enumerate(X):
        series = X[i]
        feature = feature_X[i]

        for label in range(len(centroids)):
            distance_dba, _ = fastdtw(series, centroids[label], dist=euclidean)
            max_distance_dba = max(distance_dba, max_distance_dba)  # for normalization
            min_distance_dba = min(distance_dba, min_distance_dba)
            distance_feature = euclidean(feature, features[label])
            max_distance_feature = max(distance_feature, max_distance_feature)
            min_distance_feature = min(distance_feature, min_distance_feature)

    for i, x in enumerate(X):
        min_distance = float("inf")
        classification = -1

        series = X[i]
        feature = feature_X[i]

        for label in range(len(centroids)):
            distance_dba, _ = fastdtw(series, centroids[label], dist=euclidean)
            distance_feature = euclidean(feature, features[label])
            distance_dba_normalized = (distance_dba - min_distance_dba) / (
                max_distance_dba - min_distance_dba
            )
            distance_feature_normalized = (distance_feature - min_distance_feature) / (
                max_distance_feature - min_distance_feature
            )
            distance_normalized = (
                alpha * distance_dba_normalized + (1 - alpha) * distance_feature_normalized
            )

            if distance_normalized < min_distance:
                min_distance = distance_normalized
                classification = label

        classifications.append(classification)

    return classifications


def SVM(X, labels):
    model = svm.SVC(kernel="linear")
    model.fit(X, labels)

    return model
