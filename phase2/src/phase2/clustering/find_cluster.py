import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pyclustering.cluster.center_initializer import random_center_initializer
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import random_center_initializer


def find_number_of_clusters_elbow(data, min_clusters, max_clusters):
    wcss = []  # Within-cluster sum of squares

    for i in range(min_clusters, max_clusters + 1):
        km = KMeans(n_clusters=i, init="k-means++")
        km.fit(data)
        wcss.append(km.inertia_)

    diff = np.diff(wcss)
    diff_r = diff[1:] / diff[:-1]
    optimal_clusters = np.argmin(diff_r) + min_clusters  # Adding 2 due to zero-based indexing

    return optimal_clusters


def find_number_of_clusters_silhouette(data, min_clusters, max_clusters):
    silhouette_coefficients = []

    for i in range(min_clusters, max_clusters + 1):
        km = KMeans(n_clusters=i, init="k-means++")
        clusters = km.fit_predict(data)
        silhouette_coefficients.append(silhouette_score(data, clusters))

    number_of_clusters = silhouette_coefficients.index(max(silhouette_coefficients)) + min_clusters

    return number_of_clusters


def find_number_of_clusters_gap(data, max_k):
    gaps = np.zeros(max_k)

    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k)
        km.fit(data)

        wcss = km.inertia_

        reference = np.random.random_sample(data.shape)

        reference_kmeans = KMeans(n_clusters=k)
        reference_kmeans.fit(reference)

        reference_wcss = reference_kmeans.inertia_

        gaps[k - 1] = np.log(reference_wcss) - np.log(wcss)

    max_gap_index = np.argmax(gaps)

    if max_gap_index == len(gaps) - 1:
        return max_gap_index + 1

    if gaps[max_gap_index] > gaps[max_gap_index + 1] - np.std(gaps):
        return max_gap_index + 1  # Return the best number of clusters
    else:
        return -1  # No clear best number of clusters


def find_number_of_clusters_information(data, min_k, max_k, criterion="bic"):
    ic_values = []

    for k in range(min_k, max_k + 1):
        initial_centers = random_center_initializer(data, k).initialize()

        kmeans_instance = kmeans(data, initial_centers)
        kmeans_instance.process()

        if criterion == "aic":
            ic_value = kmeans_instance.get_total_wce() + 2 * k
        elif criterion == "bic":
            ic_value = kmeans_instance.get_total_wce() + np.log(data.shape[0]) * k

        ic_values.append(ic_value)

    best_k = np.argmin(ic_values) + min_k

    return best_k
