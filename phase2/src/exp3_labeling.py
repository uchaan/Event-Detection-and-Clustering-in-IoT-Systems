import os
import argparse
import warnings
import pickle

warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from pprint import pprint

from phase2.clustering.clustering import *
from phase2.clustering.find_cluster import *
from phase2.feature.extract import *
from utils.dataset import *
from evaluation.performance import *


numbers_true = {
    "ArabicDigits": 10,
    "CharacterTrajectories": 20,
    "JapaneseVowels": 9,
    "LIBRAS": 15,
    "integrated": 3,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="integrated")

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

    with open(os.path.join(dataset_path, "pred_events_sampled_reduced.pkl"), "rb") as f:
        pred_events_sampled = pickle.load(f)

    modified_dataframes = []

    for pred_event in pred_events_sampled:
        modified_dataframes.append(pd.DataFrame(pred_event))

    X_series = dataframes_to_numpy(modified_dataframes)  # Time-series segments

    num_instances, num_timestamps, num_variables = X_series.shape

    X = dataframes_to_numpy(modified_dataframes).reshape(
        num_instances, num_timestamps * num_variables
    )

    k = 5

    total_results = {}
    total_results["kmeans_dtw"] = []
    total_results["xmeans"] = []

    for iteration in range(1):
        results = {}

        scaler = StandardScaler()

        pred_labels = xmeans_clustering(X)

        with open(os.path.join(dataset_path, "pred_labels_xmeans_sampled.pkl"), "wb") as f:
            pickle.dump(pred_labels, f, pickle.HIGHEST_PROTOCOL)

        print("iteration=", iteration)
        print()
        pprint(results)
        print()

    print(total_results)
