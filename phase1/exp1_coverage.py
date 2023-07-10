import json
import pickle
import os
import argparse

import natsort

from experiment.coverage import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lstm")
    parser.add_argument("--dataset", type=str, default="SMD")

    args = vars(parser.parse_args())

    result_root = f"./benchmark/benchmark_exp_details/{args['dataset']}/{args['model']}/{args['model']}_{args['dataset']}"
    result_name = "pred_results.pickle"

    thresholds_name = "thresholds.json"

    event_lengths = []
    count = 1

    total_cover = 0
    total_length = 0
    avg_coverage = 0

    total_cover_multi = 0
    total_length_multi = 0
    avg_coverage_multi = 0
    count_multi = 1

    total_len_of_events = 0
    total_len_of_samples = 0

    for entity in natsort.natsorted(os.listdir(result_root)):
        pred_results = {}
        thresholds = {}

        if os.path.isdir(os.path.join(result_root, entity)):
            with open(os.path.join(result_root, entity, result_name), "rb") as f:
                pred_results = pickle.load(f)

            with open(os.path.join(result_root, entity, thresholds_name), "rb") as f:
                thresholds = json.load(f)
        else:
            continue

        threshold = float(thresholds["best_adjusted"])

        true_labels = pred_results["anomaly_label"].tolist()
        pred_scores = pred_results["anomaly_score"].tolist()
        pred_labels = [1 if (score > threshold) else 0 for score in pred_scores]

        target_events = count_events(true_labels)
        pred_events = count_events(pred_labels)

        total_len_of_events += len(target_events)

        result = calculate_coverage(target_events, pred_events)

        total_cover += result["total_cover"]
        total_length += result["total_length"]
        avg_coverage += result["avg"]
        count += 1

        generated_events = multiscale_sampling(
            pred_events, pred_scores, threshold, alpha=0.01, extend_scale=1.5
        )

        total_len_of_samples += len(generated_events)

        result_multi = calculate_coverage(target_events, generated_events)

        total_cover_multi += result_multi["total_cover"]
        total_length_multi += result_multi["total_length"]
        avg_coverage_multi += result_multi["avg"]
        count_multi += 1

    print(total_len_of_events)
    print(total_len_of_samples)

    print()
    print("** Total Result **")
    print("total coverage of pred events=", total_cover / total_length)
    print("total average of pred events=", avg_coverage / count)

    print("total coverage of generated events=", total_cover_multi / total_length_multi)
    print("total average of generated events=", avg_coverage_multi / count_multi)
