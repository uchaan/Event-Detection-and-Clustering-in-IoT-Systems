from tsfresh import extract_features
import numpy as np
from scipy.stats import skew, kurtosis
from phase2.t2f.extractor import feature_extraction
from phase2.t2f.importance import feature_selection


def extract_t2f(X, batch_size):
    features = feature_extraction(X, batch_size=batch_size, p=1)

    transform_type = "std"  # preprocessing step
    model_type = "KMeans"  # clustering model

    context = {"model_type": model_type, "transform_type": transform_type}

    top_feats = feature_selection(features, labels={}, context=context)

    features = features[top_feats]

    return features


def extract_tsfresh(dataframes):
    extracted_features = extract_features(
        dataframes,
        column_id="segment_id",
    )

    return extracted_features


def extract_tsfresh_each(dataframes):
    result = []

    for df in dataframes:
        features = extract_features(df, column_id="segment_id")
        result.append(features)

    return result


def extract_statistic(segment):
    """
    features:
        length
        mean (each dim)
        median (each dim)
        standard dev. (each dim)
        min (each dim)
        max (each dim)
        skewness (each dim)
        kurtosis (each dim)
    """

    features = []
    segment_array = segment.values

    length = len(segment)
    features.append(length)

    means = np.mean(segment_array, axis=0)
    features += list(means)

    median = np.median(segment_array, axis=0)
    features += list(median)

    std = np.std(segment_array, axis=0)
    features += list(std)

    minimum = np.min(segment_array, axis=0)
    features += list(minimum)

    maximum = np.max(segment_array, axis=0)
    features += list(maximum)

    skewness = skew(segment_array, axis=0)
    features += list(skewness)

    kurto = kurtosis(segment_array, axis=0)
    features += list(kurto)

    return features


def extract_autocorrelation(segment):
    result = []

    for col in segment.columns:
        series = segment[col]
        autocorrelation = series.autocorr()
        result.append(autocorrelation)

    return autocorrelation
