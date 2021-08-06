import pandas as pd
import numpy as np
import json



def normalise(dataset_path, save_path=None, w_map_path=None, h_map_path=None):
    average_widths = {}
    average_heights = {}

    dataset = pd.read_csv(dataset_path)
    X = dataset.iloc[:, 8:].values
    corner_x = dataset.iloc[:, 4].values
    corner_y = dataset.iloc[:, 5].values
    widths = dataset.iloc[:, 6].values
    heights = dataset.iloc[:, 7].values
    classes = dataset.iloc[:, 2].values

    for cls in dataset.iloc[:, 2].unique():
        indices = [i for i in range(0, len(classes)) if classes[i] == cls]

        average_widths[cls] = np.mean(widths[indices])
        average_heights[cls] = np.mean(heights[indices])

    new_widths = np.asarray([average_widths[c] for c in classes ])
    new_heights = np.asarray([average_heights[c] for c in classes ])


    for i in range(136):
        X[:, i] = (X[:, i] - corner_x) * (new_widths / widths)
        X[:, i+1] = (X[:, i+1] - corner_y) * (new_heights / heights)

    dataset.iloc[:, 8:] = X
    dataset.iloc[:, 6] = new_widths
    dataset.iloc[:, 7] = new_heights

    if w_map_path is not None:
        with open(w_map_path, "w") as f:
            json.dump(average_widths, f)
    if h_map_path is not None:
        with open(h_map_path, "w") as f:
            json.dump(average_heights, f)
    if save_path is not None:
    	dataset.to_csv(save_path, index=False)
    else:
    	return dataset


normalise("../yadav_dataset.csv", "yadav_normalised_dataset.csv", w_map_path="width_mapping.json", h_map_path="height_mapping.json")
