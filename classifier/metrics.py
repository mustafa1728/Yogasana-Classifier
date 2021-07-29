from sklearn.metrics import classification_report
import pandas as pd
import json
import joblib


predictions = pd.read_csv("predictions_frame_wise.csv")
with open("ids_to_class.json") as f:
    id_to_class_mapping = json.load(f)
classifier = joblib.load("models/model_subsampled.z")
labels = [id_to_class_mapping[str(c)] for c in classifier.classes_]

Y = predictions.iloc[:, 0].values
Y_hat = predictions.iloc[:, 1].values
mask = [i for i in range(len(Y)) if Y_hat[i] != -1]

Y = Y[mask]
Y_hat = Y_hat[mask]
print(labels)

report = classification_report(Y, Y_hat, target_names=labels, output_dict = True)
# report = classification_report(Y, Y_hat)
# print(type(report))
# print(report)
df = pd.DataFrame(report)
df = df.transpose()
df.to_csv("subject_wise_metric.csv")