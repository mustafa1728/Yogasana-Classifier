import pandas as pd
import os
import json
from classifier.model import Classifier
import numpy as np
from sklearn.metrics import top_k_accuracy_score

def generate_dataset(root_data_dir, save_path):

    final_dict = {"image_path": [], "class": []}
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
            for j in range(136):
                for k in range(3):
                    final_dict["keypt" + "_" + str(j) + "_" + str(k)].append(kps[j*3 + k])
    df = pd.DataFrame(final_dict)
    df.to_csv(save_path, index=False)

def preprocess_labels(Y1, Y2):
    Y = np.asarray(Y1.tolist() + Y2.tolist())
    classes = np.unique(Y)
    class_to_id_mapping = {}
    id_to_class_mapping = {}
    for i in range(len(classes)):
        id_to_class_mapping[str(i)] = classes[i]
        class_to_id_mapping[classes[i]] = i
    Y1_processed = [class_to_id_mapping[i] for i in Y1]
    Y2_processed = [class_to_id_mapping[i] for i in Y2]
    with open("yoga_82_id_to_class.json", "w") as f:
        json.dump(id_to_class_mapping, f)
    return Y1_processed, Y2_processed
            
def experiment(dataset_path, train_idx_path, test_idx_path):
    data = pd.read_csv(dataset_path)
    train_paths = pd.read_csv(train_idx_path, sep=',', lineterminator='\n', header = None)
    test_paths = pd.read_csv(test_idx_path, sep=',', lineterminator='\n', header = None)
    train_paths = train_paths.iloc[:, 0].values
    test_paths = test_paths.iloc[:, 0].values

    paths = data.iloc[:, 0].values
    Y = data.iloc[:, 1].values
    X = data.iloc[:, 2:].values

    print(paths[:5])
    print(train_paths[:5])

    mask_train = np.array([pth in train_paths for pth in paths])
    mask_test = np.array([pth in test_paths for pth in paths])

    X_train, Y_train = X[mask_train], Y[mask_train]
    X_test, Y_test = X[mask_test], Y[mask_test]

    print(X_train.shape, X_test.shape)

    Y_train, Y_test = preprocess_labels(Y_train, Y_test)

    classifier = Classifier("random_forest", no_estimators = 500, max_depth = None)
    X_train, X_test = classifier.scale_vectors(X_train, X_test)
    classifier.train(X_train, Y_train)

    Y_pred = classifier.model.predict_proba(X_test)
    accuracy = classifier.model.score(X_test, Y_test)
    top1_accuracy = top_k_accuracy_score(Y_test, Y_pred, k=1)
    top5_accuracy = top_k_accuracy_score(Y_test, Y_pred, k=5)

    print("Normal Accuracy on Yoga-82: {:.2f}%".format(accuracy*100))
    print("Top 1 Accuracy on Yoga-82: {:.2f}%".format(top1_accuracy*100))
    print("Top 5 Accuracy on Yoga-82: {:.2f}%".format(top5_accuracy*100))




if __name__ == "__main__":
    # generate_dataset(root_data_dir = "yoga_82_alphapose_kps", save_path = "yoga_82_complete.csv")
    experiment("yoga_82_complete.csv", "/Users/mustafa/Desktop/Yoga-82/yoga_train.txt", "/Users/mustafa/Desktop/Yoga-82/yoga_test.txt")