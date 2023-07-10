import pandas as pd
import csv
import pickle


info = pd.read_csv("./types.csv")

window_size = 100

dataframes = []
events = []
labels = []

label_dict = {"metric": 0, "temporal": 1, "metric-temporal": 2}
max_size = 0

with open("./types.csv", "r", encoding="cp949") as f:
    reader = csv.reader(f)
    header = next(reader)
    print(header)

    for row in reader:
        print(row)
        (entity, index, label) = row
        fname = f"{entity}_test.pkl"
        start, end = index.split("-")
        start = int(start)
        end = int(end)
        start -= window_size

        max_size = max(end + 1 - start, max_size)

        with open(f"./data/ASD/{fname}", "rb") as pkl:
            data = pickle.load(pkl)
            series = data[start : end + 1]

            df = pd.DataFrame(series)

            df["label"] = label_dict[label]

            events.append(series)
            labels.append(label_dict[label])

            dataframes.append(df)

        print(fname)
        print(start)
        print(end)

print(events)
print(len(events))
print(labels)
print(len(labels))
print(max_size)
