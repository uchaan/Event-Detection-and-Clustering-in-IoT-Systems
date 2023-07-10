import numpy as np
from fastdtw import fastdtw
from tqdm import tqdm
from tslearn.barycenters import dtw_barycenter_averaging


def dtw_distance(ts1, ts2):
    distance, _ = fastdtw(ts1, ts2)
    return distance


def align_time_series(ts1, ts2):
    n, m = len(ts1), len(ts2)
    alignment = np.zeros((n, m))

    alignment[0][0] = dtw_distance(ts1[0], ts2[0])

    for i in range(1, n):
        alignment[i][0] = alignment[i - 1][0] + dtw_distance(ts1[i], ts2[0])

    for j in range(1, m):
        alignment[0][j] = alignment[0][j - 1] + dtw_distance(ts1[0], ts2[j])

    for i in range(1, n):
        for j in range(1, m):
            alignment[i][j] = min(
                alignment[i - 1][j], alignment[i][j - 1], alignment[i - 1][j - 1]
            ) + dtw_distance(ts1[i], ts2[j])

    return alignment


def dba(time_series, max_iterations=10, convergence_threshold=0.01):
    n, ts_length, num_variables = time_series.shape
    barycenter = np.copy(time_series[0])

    for _ in tqdm(range(max_iterations)):
        previous_barycenter = np.copy(barycenter)

        alignment_sum = np.zeros_like(barycenter)
        counts = np.zeros((ts_length, num_variables))

        for i in range(n):
            alignment = align_time_series(barycenter, time_series[i])

            for j in range(ts_length):
                for k in range(num_variables):
                    alignment_sum[j][k] += time_series[i][j][k] / alignment[j][ts_length - 1]
                    counts[j][k] += 1 / alignment[j][ts_length - 1]

        barycenter = alignment_sum / counts

        if np.linalg.norm(barycenter - previous_barycenter) < convergence_threshold:
            break

    return barycenter


def DBA(time_series, max_iterations=10):
    return dtw_barycenter_averaging(time_series, max_iter=max_iterations)
