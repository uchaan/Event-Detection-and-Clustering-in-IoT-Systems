import os
import sys
import gc
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

from phase1.anomaly_detection.utils import *
from phase1.anomaly_detection.usad import *
from visualization.visualize import *

# Parameter Settings

events_path = "../data/swat/events.json"
normal_path = "../data/swat/SWaT_Dataset_Normal_v1.csv"
attack_path = "../data/swat/SWaT_Dataset_Attack_v0.csv"
model_path = "./models/"
model_name = "usad_model_6.pth"

window_size = 20  # step size is 1
BATCH_SIZE = 7919
N_EPOCHS = 100
hidden_size = 100
scale = "standard"
# scale = "minmax"


if __name__ == "__main__":
    print()
    print("** Current Machine Information **")
    os.system("nvidia-smi -L")
    print()

    ## Attack Dataset
    # Read data
    attack = pd.read_csv(attack_path, sep=";")  # , nrows=1000)
    labels = [float(label != "Normal") for label in attack["Normal/Attack"].values]
    attack = attack.drop(["Timestamp", "Normal/Attack"], axis=1)

    # Transform all columns into float64
    for i in list(attack):
        attack[i] = attack[i].apply(lambda x: str(x).replace(",", "."))
    attack = attack.astype(float)

    print()
    print("** Original attack dataset **")
    print(attack.head(3))
    print()
    print("** Shape of attack dataset **")
    print(attack.shape)
    print()

    ## Visualize Dataset
    # Normal data
    features_considered = ["LIT301", "AIT201", "AIT202", "AIT203"]
    features = attack[features_considered]
    # features.index = attack["Timestamp"]
    # plot_anomaly("True Data", features, labels, 1)

    events = count_events(labels)
    print(events)

    with open(events_path, "w") as f:
        json.dump(events, f, indent=4)

    # print(json_list)
