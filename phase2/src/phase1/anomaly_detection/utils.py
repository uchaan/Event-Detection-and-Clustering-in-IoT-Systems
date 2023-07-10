import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from sklearn.metrics import roc_curve, roc_auc_score


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def plot_history(history):
    losses1 = [x["val_loss1"] for x in history]
    losses2 = [x["val_loss2"] for x in history]
    plt.plot(losses1, "-x", label="loss1")
    plt.plot(losses2, "-x", label="loss2")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Losses vs. No. of epochs")
    plt.grid()
    plt.show()


def histogram(y_test, y_pred):
    plt.figure(figsize=(12, 6))
    plt.hist(
        [y_pred[y_test == 0], y_pred[y_test == 1]],
        bins=20,
        color=["#82E0AA", "#EC7063"],
        stacked=True,
    )
    plt.title("Results", size=20)
    plt.grid()
    plt.show()


def ROC(y_test, y_pred):
    fpr, tpr, tr = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    idx = np.argwhere(np.diff(np.sign(tpr - (1 - fpr)))).flatten()

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.plot(fpr, tpr, label="AUC=" + str(auc))
    plt.plot(fpr, 1 - fpr, "r:")
    plt.plot(fpr[idx], tpr[idx], "ro")
    plt.legend(loc=4)
    plt.grid()
    plt.show()
    return tr[idx]


def confusion_matrix(target, predicted, perc=False):
    data = {"y_Actual": target, "y_Predicted": predicted}
    df = pd.DataFrame(data, columns=["y_Predicted", "y_Actual"])
    confusion_matrix = pd.crosstab(
        df["y_Predicted"], df["y_Actual"], rownames=["Predicted"], colnames=["Actual"]
    )

    if perc:
        sns.heatmap(
            confusion_matrix / np.sum(confusion_matrix), annot=True, fmt=".2%", cmap="Blues"
        )
    else:
        sns.heatmap(confusion_matrix, annot=True, fmt="d")
    plt.show()


def count_events(labels):
    events = []
    event_count = 0
    current_length = 0
    prev = "normal"
    start = 0
    end = 0

    for i, label in enumerate(labels):
        if label == 0:  # normal
            if prev == "attack":
                end = i
                events.append({"start": start, "end": end, "len": current_length})
                start = 0
                end = 0
                current_length = 0
            prev = "normal"
        else:  # attack
            current_length += 1
            if prev == "normal":
                prev = "attack"
                event_count += 1
                start = i

    return events


def calculate_coverage(target_events, pred_events):
    # print(target_events)
    acc_coverage = 0
    total_cover = 0
    total_length = 0
    for i, target_event in enumerate(target_events):
        target_start = target_event["start"]
        target_end = target_event["end"]
        target_length = target_event["len"]

        best_cover = 0

        for j, pred_event in enumerate(pred_events):
            pred_start = pred_event["start"]
            pred_end = pred_event["end"]
            pred_length = pred_event["len"]

            if pred_end <= target_start or pred_start >= target_end:
                continue
            elif pred_start <= target_start and pred_end >= target_end:
                coverage = pred_start - pred_end - 2 * target_start + 2 * target_end
            elif pred_start >= target_start and pred_end <= target_end:
                coverage = pred_end - pred_start
            else:
                if target_start >= pred_start:
                    coverage = pred_start + pred_end - 2 * target_start
                else:
                    coverage = 2 * target_end - pred_start - pred_end
            best_cover = max(best_cover, coverage)

        # print(best_cover / (target_end - target_start))
        acc_coverage += best_cover / (target_end - target_start)
        total_cover += best_cover
        total_length += target_length

    result = {
        "avg": acc_coverage / len(target_events),
        "total_cover": total_cover,
        "total_length": total_length,
    }

    return result
