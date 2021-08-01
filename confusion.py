import pandas as pd
from sklearn.metrics import confusion_matrix 
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np

# predictions = pd.read_csv("classifier/predictions_frame_wise.csv")
predictions = pd.read_csv("classifier/predictions_camera_wise_1cam.csv")


Y = predictions.iloc[:, 0].values
Y_pred = predictions.iloc[:, 1].values
print(len(Y_pred))
labels = np.unique(Y_pred)

matrix = confusion_matrix(Y, Y_pred, labels=labels)
print(matrix)
print(matrix.shape)

df_cm = pd.DataFrame(matrix, range(12), range(12))

sn.set(font_scale=1.4)
fig, ax = plt.subplots(figsize=(15, 12))
cfsion = sn.heatmap(df_cm, annot=True, annot_kws={"size": 20}, cmap=plt.cm.Blues,ax = ax, fmt="d")
cfsion.set_yticklabels(cfsion.get_ymajorticklabels(), fontsize = 36)
cfsion.set_xticklabels(cfsion.get_ymajorticklabels(), fontsize = 36)

# plt.show()
# plt.savefig("confusion.png")
plt.savefig("confusion_camera.png")