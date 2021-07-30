from sklearn.metrics import classification_report
import pandas as pd
import json
import joblib
import numpy as np

def predictions_to_df(prediction_path):
    predictions = pd.read_csv(prediction_path)
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
    df = pd.DataFrame(report)
    df = df.transpose()
    # df.to_csv("camera_wise_metric_2cam_1.csv")
    return df

# df0 = predictions_to_df("predictions_camera_wise_3cam3cam_0.csv")
# df1 = predictions_to_df("predictions_camera_wise_3cam3cam_1.csv")
# df2 = predictions_to_df("predictions_camera_wise_3cam3cam_2.csv")
# df3 = predictions_to_df("predictions_camera_wise_3cam3cam_3.csv")

df0 = predictions_to_df("predictions_camera_wise2cam_0.csv")
df1 = predictions_to_df("predictions_camera_wise2cam_1.csv")
df2 = predictions_to_df("predictions_camera_wise2cam_2.csv")
df3 = predictions_to_df("predictions_camera_wise2cam_3.csv")


new_df = pd.DataFrame(index = df0.index, columns = df0.columns)

for col in new_df.columns:
    stack = np.vstack([df0[col], df1[col], df2[col], df3[col]])
    print(stack)
    new_var = np.mean(stack, axis = 0)
    new_df[col] = new_var

new_df.to_csv("metrics_2cam.csv")

    


