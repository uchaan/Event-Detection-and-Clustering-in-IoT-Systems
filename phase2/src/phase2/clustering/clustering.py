import numpy as np
from tslearn.clustering import TimeSeriesKMeans, KShape, KernelKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_distances
from pyclustering.cluster.xmeans import xmeans


def kmeans_clustering_tslearn_dtw(X, k, scale=False):
    if scale:
        scaler = TimeSeriesScalerMeanVariance()
        X = scaler.fit_transform(X)

    kmeans = TimeSeriesKMeans(n_clusters=k, metric="dtw")
    kmeans.fit(X)

    labels = kmeans.labels_

    return labels


def kshape_clustering_tslearn(X, k, scale=False):
    if scale:
        scaler = TimeSeriesScalerMeanVariance()
        X = scaler.fit_transform(X)

    kshape = KShape(n_clusters=k)
    kshape.fit(X)

    labels = kshape.labels_

    return labels


def kernelkmeans_clustering_tslearn(X, k, scale=False):
    if scale:
        scaler = TimeSeriesScalerMeanVariance()
        X = scaler.fit_transform(X)

    kkmeans = KernelKMeans(n_clusters=k)
    kkmeans.fit(X)

    labels = kkmeans.labels_

    return labels


def kmeans_clustering_tslearn_softdtw(X, k, scale=False):
    if scale:
        scaler = TimeSeriesScalerMeanVariance()
        X = scaler.fit_transform(X)

    kmeans = TimeSeriesKMeans(n_clusters=k, verbose=False, metric="softdtw")
    kmeans.fit(X)

    labels = kmeans.labels_

    return labels


def kmeans_clustering_tslearn_euclidean(X, k, scale=False):
    if scale:
        scaler = TimeSeriesScalerMeanVariance()
        X = scaler.fit_transform(X)

    kmeans = TimeSeriesKMeans(n_clusters=k, verbose=False, metric="euclidean")
    kmeans.fit(X)

    labels = kmeans.labels_

    return labels


def kmeans_clustering_vectors(X, k):
    X[np.isnan(X)] = 0

    kmeans = KMeans(n_clusters=k)

    labels = kmeans.fit_predict(X)

    return labels


def agglomerative_clustering_num(X, max_k=100, scale=False):
    if scale:
        scaler = TimeSeriesScalerMeanVariance()
        X = scaler.fit_transform(X)

    agg_clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=3.0)
    agg_clustering.fit(X)

    labels = agg_clustering.labels_
    num_clusters = np.max(labels) + 1

    return num_clusters


def agglomerative_clustering(X, max_k=100, scale=False):
    if scale:
        scaler = TimeSeriesScalerMeanVariance()
        X = scaler.fit_transform(X)

    agg_clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=3.0)
    agg_clustering.fit(X)

    labels = agg_clustering.labels_

    return labels


def divisive_clustering_num(X, threshold):
    dist_matrix = pairwise_distances(X)

    num_segments = len(dist_matrix)
    labels = np.zeros(num_segments)

    def divide_clusters(dist_matrix, labels, threshold):
        max_dist = np.max(dist_matrix)

        if max_dist < threshold:
            return

        idx = np.argmax(dist_matrix)
        i, j = np.unravel_index(idx, dist_matrix.shape)
        labels[j] = np.max(labels) + 1

        mask = labels == labels[j]
        dist_matrix[mask, mask] = max_dist + 1

        divide_clusters(dist_matrix[:j, :j], labels[:j], threshold)
        divide_clusters(dist_matrix[j + 1 :, j + 1 :], labels[j + 1 :], threshold)

    cluster_labels = divide_clusters(dist_matrix, labels, threshold)

    num_clusters = len(np.unique(cluster_labels))

    return num_clusters


def xmeans_clustering_num(X, max_k):
    xmeans_instance = xmeans(X, kmax=max_k, initial_clusters=2)
    xmeans_instance.process()

    labels = xmeans_instance.get_clusters()
    num_clusters = len(np.unique(labels))

    return num_clusters


def xmeans_clustering(X):
    xmeans_instance = xmeans(X, kmax=30, initial_clusters=2)
    xmeans_instance.process()

    labels = xmeans_instance.get_clusters()

    predicted_labels = np.zeros(len(X), dtype=int)
    for cluster_idx, cluster in enumerate(labels):
        predicted_labels[cluster] = cluster_idx

    return predicted_labels


def dbscan_num(X):
    dbscan = DBSCAN(eps=1.0, min_samples=5)
    dbscan.fit(X)

    labels = dbscan.labels_
    num_clusters = len(np.unique(labels)) - 1

    return num_clusters


def dbscan(X):
    dbscan = DBSCAN(eps=1.0, min_samples=5)
    dbscan.fit(X)

    labels = dbscan.labels_

    return labels


def optics_num(X):
    optics = OPTICS(min_samples=5, xi=0.05)
    optics.fit(X)

    labels = optics.labels_
    num_clusters = len(np.unique(labels)) - 1

    return num_clusters


def optics(X):
    optics = OPTICS(min_samples=5, xi=0.05)
    optics.fit(X)

    labels = optics.labels_

    return labels
