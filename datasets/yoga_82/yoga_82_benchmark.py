from operator import index
from numpy.lib.npyio import save
import pandas as pd
import os
import json
from classifier.model import Classifier
import numpy as np
from sklearn.metrics import top_k_accuracy_score
from sklearn.model_selection import StratifiedKFold
from classifier.utils import AccuracyMeter

def generate_data(root_data_dir, save_path):

    final_dict = {"image_path": [], "class": [], "x0": [], "y0": [], "width": [], "height": []}
    for i in range(136):
        for j in range(3):
            final_dict["keypt" + "_" + str(i) + "_" + str(j)] = []

    folders_list = [x[0] for x in os.walk(root_data_dir)]
    folders_list = folders_list[1:]
    folders_list = [fol.split('/')[-1] for fol in folders_list]

    for foldername in folders_list:
        label = foldername
        folder_path = os.path.join(root_data_dir, foldername)
        json_path = os.path.join(folder_path, "alphapose-results.json")
        try:
            with open(json_path) as f:
                key_points_data = json.load(f)
        except:
            continue

        for kp_dict in key_points_data:
            img_path = os.path.join(foldername, kp_dict["image_id"])
            kps = kp_dict["keypoints"]
            final_dict["image_path"].append(img_path)
            final_dict["class"].append(label)
            x0, y0, w, h = kp_dict["box"][0], kp_dict["box"][1], kp_dict["box"][2], kp_dict["box"][3]
            final_dict["x0"].append(x0)
            final_dict["y0"].append(y0)
            final_dict["width"].append(w)
            final_dict["height"].append(h)
            for j in range(136):
                for k in range(3):
                    final_dict["keypt" + "_" + str(j) + "_" + str(k)].append(kps[j*3 + k])
    df = pd.DataFrame(final_dict)
    df.to_csv(save_path, index=False)

def process_dataset_to_get_largest_bbox(dataset_path, save_path):
    dataset = pd.read_csv(dataset_path)
    print(dataset.describe())
    dataset["size"] = dataset["width"].values * dataset["height"].values
    ids_max_bbox = dataset.groupby(['image_path'])["size"].idxmax()
    dataset = dataset.iloc[ids_max_bbox, :]
    dataset = dataset.drop(["size"], axis = 1)
    print(dataset.describe())
    dataset.to_csv(save_path, index=False)

def preprocess_labels(Y1, Y2):
    Y = np.asarray(Y1.tolist() + Y2.tolist())
    classes = np.unique(Y)
    class_to_id_mapping = {}
    id_to_class_mapping = {}
    for i in range(len(classes)):
        id_to_class_mapping[str(i)] = classes[i]
        class_to_id_mapping[classes[i]] = i
    print(id_to_class_mapping)
    Y1_processed = [class_to_id_mapping[i] for i in Y1]
    Y2_processed = [class_to_id_mapping[i] for i in Y2]
    with open("yoga_82_id_to_class.json", "w") as f:
        json.dump(id_to_class_mapping, f)
    return Y1_processed, Y2_processed

def check_curated(paths, curated_root_dir = "/Users/mustafa/Desktop/yoga/Yoga-82/curated_data"):
    curated_paths = [p for p in paths if os.path.isfile(os.path.join(curated_root_dir, p))]
    print("curation: {} -> {}".format(len(paths), len(curated_paths)))
    return curated_paths
    

def get_Y(data, train_df, test_df):
    paths = data.iloc[:, 0].values
    Y = []
    train_paths = train_df.iloc[:, 0].values
    test_paths = test_df.iloc[:, 0].values
    for pth in paths:
        found = False
        for i in range(train_paths.shape[0]):
            if train_paths[i] == pth:
                Y.append(train_df.iloc[i, 3])
                found = True
        if not found:
            for i in range(test_paths.shape[0]):
                if test_paths[i] == pth:
                    Y.append(test_df.iloc[i, 3])
                    found = True
    return np.asarray(Y)

def experiment(data_path, train_idx_path, test_idx_path, level=3):
    print("Experimenting for level {}".format(level))
    data = pd.read_csv(data_path)
    train_df = pd.read_csv(train_idx_path, sep=',', lineterminator='\n', header = None)
    test_df = pd.read_csv(test_idx_path, sep=',', lineterminator='\n', header = None)
    train_paths = train_df.iloc[:, 0].values
    test_paths = test_df.iloc[:, 0].values

    paths = data.iloc[:, 0].values
    # Y = data.iloc[:, 1].values
    Y = get_Y(data, train_df, test_df)
    X = data.iloc[:, 6:].values
    bbox = data.iloc[:, 4:6].values

    corner_x = data.iloc[:, 2].values
    corner_y = data.iloc[:, 3].values
    widths = data.iloc[:, 4].values
    heights = data.iloc[:, 5].values
    classes = Y
    average_widths = average_heights = {}
    for cls in classes:
        indices = [i for i in range(0, len(classes)) if classes[i] == cls]

        average_widths[cls] = np.mean(widths[indices])
        average_heights[cls] = np.mean(heights[indices])

    new_widths = np.asarray([average_widths[c] for c in classes ])
    new_heights = np.asarray([average_heights[c] for c in classes ])

    for i in range(136):
        X[:, 2*i] = (X[:, 2*i] - corner_x) * (1 / widths)
        X[:, 2*i+1] = (X[:, 2*i+1] - corner_y) * (1 / heights)

    as_ratios = bbox[:, 1] / bbox[:, 0]
    as_ratios = as_ratios.reshape(-1, 1)
    # X = np.append(X, bbox, axis = 1)
    X = np.append(X, as_ratios, axis = 1)

    print(paths[:5])
    print(train_paths[:5])

    train_paths = check_curated(train_paths)
    test_paths = check_curated(test_paths)
    mask_train = np.array([pth in train_paths for pth in paths])
    mask_test = np.array([pth in test_paths for pth in paths])

    

    X_train, Y_train = X[mask_train], Y[mask_train]
    X_test, Y_test = X[mask_test], Y[mask_test]

    print(X_train.shape, X_test.shape)

    # Y_train, Y_test = preprocess_labels(Y_train, Y_test)

    classifier = Classifier("random_forest", no_estimators = 500, max_depth = None)
    # X_train, X_test = classifier.scale_vectors(X_train, X_test)
    # classifier.train(X_train, Y_train)

    # Y_pred = classifier.model.predict_proba(X_test)
    # accuracy = classifier.model.score(X_test, Y_test)
    # top1_accuracy = top_k_accuracy_score(Y_test, Y_pred, k=1)
    # top5_accuracy = top_k_accuracy_score(Y_test, Y_pred, k=5)

    # print("Normal Accuracy on Yoga-82: {:.2f}%".format(accuracy*100))
    # print("Top 1 Accuracy on Yoga-82: {:.2f}%".format(top1_accuracy*100))
    # print("Top 5 Accuracy on Yoga-82: {:.2f}%".format(top5_accuracy*100))

    print("starting 10 fold cross val")
    curated_paths = check_curated(paths)
    curated_mask = [i for i in range(len(paths)) if paths[i] in curated_paths]
    X_curated, Y_curated = X[curated_mask], Y[curated_mask]
    # Y_curated, _ = preprocess_labels(Y_curated, np.asarray([]))
    if level == 1:
        with open("/Users/mustafa/Desktop/yoga/Yoga-82/level_3_to_1.json") as f:
            level_3_to_1 = json.load(f)
        Y_curated = [level_3_to_1[str(i)] for i in Y_curated]
    elif level == 2:
        with open("/Users/mustafa/Desktop/yoga/Yoga-82/level_3_to_2.json") as f:
            level_3_to_2 = json.load(f)
        Y_curated = [level_3_to_2[str(i)] for i in Y_curated]
    else:
        if level > 3 or level < 1:
            raise ValueError("Level can be one of [1, 2, 3]. Received {}".format(level))
    Y_curated = np.asarray(Y_curated)
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2)
    predictions = np.asarray([-1 for i in range(len(Y_curated))])
    accuracy_meter = AccuracyMeter()
    for train_index, test_index in kf.split(X_curated, Y_curated):
        X_train, X_test = X_curated[train_index], X_curated[test_index]
        Y_train, Y_test = Y_curated[train_index], Y_curated[test_index]
        print(X_train.shape, X_test.shape)

        classifier = Classifier("random_forest", max_depth = None, no_estimators = 500)

        X_train, X_test = classifier.scale_vectors(X_train, X_test, "scaler.pkl")
        classifier.train(X_train, Y_train)
        accuracy = classifier.evaluate(X_test, Y_test, confusion_path = "yoga_82_confusion.png")

        predictions[test_index] = classifier.model.predict(X_test)

        # md = merge_dicts(md, metric_dict)
        print("Accuracy: {}%".format(100*accuracy))

        is_best = accuracy_meter.update(accuracy)
        if is_best:
           classifier.save_model("yoga_82_model.z")

        df_predictions = pd.DataFrame({"labels": Y_curated, "predictions": predictions})
        df_predictions.to_csv("yoga_82_asr_maxbbox_curated_predictions_level_{}.csv".format(level), index = False)

    accuracy_meter.display()


if __name__ == "__main__":
    # generate_data(root_data_dir = "yoga_82_alphapose_kps", save_path = "yoga_82_complete.csv")
    experiment("yoga_82_max_bbox_kponly.csv", "/Users/mustafa/Desktop/yoga/Yoga-82/yoga_train.txt", "/Users/mustafa/Desktop/yoga/Yoga-82/yoga_test.txt", level = 1)
    # process_dataset_to_get_largest_bbox("yoga_82_complete.csv", "yoga_82_max_bbox.csv")
    # print("this is with actual bbox!")
