from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import json
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import plot_confusion_matrix
import pandas as pd
import pickle

def load_model(model_weights_path):
    classifier = joblib.load(model_weights_path)
    return classifier


def sub_sample(X, Y, class_counts, subjects=None, cameras = None):
    no_samples_per_class = 6000
    classes = [cls for cls in list(class_counts.keys())[:12] ]

    rng = np.random.default_rng(1)
    X_subset_list = []
    Y_subset_list = []

    subj_subset_list = []
    cam_subset_list = []

    for cls in classes:
        total_samples = X[Y==cls]
        total_labels = Y[Y==cls]
        idx = rng.choice(total_samples.shape[0], size = no_samples_per_class, replace = False)
        
        X_subset_list.append(total_samples[idx])
        Y_subset_list.append(total_labels[idx])

        total_subjs = subjects[Y==cls]
        total_cams = cameras[Y==cls]
        subj_subset_list.append(total_subjs[idx])
        cam_subset_list.append(total_cams[idx])

    X_subset = np.concatenate(X_subset_list, axis = 0)
    Y_subset = np.concatenate(Y_subset_list, axis = 0)

    subj_subset = np.concatenate(subj_subset_list, axis = 0)
    cam_subset = np.concatenate(cam_subset_list, axis = 0)
    return X_subset, Y_subset, classes, subj_subset, cam_subset

def pre_process_labels(dataset, save_mapping_path = None):
    class_to_id_mapping = {}
    id_to_class_mapping = {}
    dataset["class"] = dataset["class"].apply(lambda x: still_left_to_still(x))
    dataset["class"] = dataset["class"].apply(lambda x: condition(x))
    classes = list(dataset["class"].unique())
    for i in range(len(classes)):
        class_to_id_mapping[classes[i]] = i
        id_to_class_mapping[i] = classes[i]
    dataset["class"] = dataset["class"].apply(lambda x:class_to_id_mapping[x])
    if save_mapping_path is not None:
        with open(save_mapping_path, 'w') as f:
            json.dump(id_to_class_mapping, f)
    return dataset

def still_left_to_still(x):
    if x == "Still_left": 
        return "Still" 
    else:  
        return x

def condition(x):
    if x == "None": 
        return "Still" 
    else:  
        return x

def get_dataset(dataset_path, include_as_ratio = False, return_subj = False, return_cams = False):
    dataset = pd.read_csv(dataset_path)
    dataset.dropna(inplace=True)
    indices_to_keep = ~dataset.isin([np.nan, np.inf, -np.inf]).any(1)
    dataset = dataset[indices_to_keep]
    dataset = pre_process_labels(dataset)
    X = dataset.iloc[:, 8:].values
    Y = dataset.iloc[:, 3].values
    if include_as_ratio:
        bbox = dataset.iloc[:, 6:8].values
        as_ratios = bbox[:, 1] / bbox[:, 0]
        as_ratios = as_ratios.reshape(-1, 1)
        X = np.append(X, as_ratios, axis = 1)
    X, Y, classes, subj_subset, cam_subset = sub_sample(X, Y, dataset["class"].value_counts().to_dict(), dataset["subject"].values, dataset["camera"].values)
    if return_subj:
        if return_cams:
            return X, Y, subj_subset, cam_subset
        else:
            return X, Y, subj_subset
    elif return_cams:
        return X, Y, cam_subset
    else:
        return X, Y

def save_confusion(classifier, X_Test, Y_Test, display_labels=None, save_path = "confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(15, 12))
    plot_confusion_matrix(
        classifier, 
        X_Test, Y_Test, ax=ax,
        display_labels=display_labels, 
        cmap=plt.cm.Blues,
        normalize="pred",
        xticks_rotation = "vertical"
    )
    plt.savefig(save_path, dpi = 300)

def create_train_test(X, Y, test_size):
    # Splitting the dataset into the Training set and Test set
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = test_size, random_state = 0)
    
    # Feature Scaling
    print("mean: ", np.mean(X_Train, axis = 0))
    print("std: ", np.std(X_Train, axis = 0), np.var(X_Train, axis = 0))
    sc_X = StandardScaler()
    X_Train = sc_X.fit_transform(X_Train)
    X_Test = sc_X.transform(X_Test)

    with open("scalar.pkl", "wb") as f:
        pickle.dump(sc_X, f)
    print("scalar's mean: ", sc_X.mean_)
    print("scalar's std: ", sc_X.var_, np.sqrt(sc_X.var_))


    return X_Train, X_Test, Y_Train, Y_Test

def predict_class(input_features, model_weights_path = "model.z", saved_mapping = "ids_to_class.json"):
    model = load_model(model_weights_path)
    class_pred = model.predict([input_features])
    with open(saved_mapping) as f:
        id_to_class_mapping = json.load(f)
    return id_to_class_mapping[class_pred]

class AccuracyMeter():
    def __init__(self):
        self.best_accuracy = None
        self.worst_accuracy = None
        self.average_accuracy = 0
        self.count = 0

    def update(self, accuracy):
        self.average_accuracy = (self.average_accuracy*self.count + accuracy) / (self.count+1)
        self.count += 1
        if self.worst_accuracy is None or accuracy<=self.worst_accuracy:
            self.worst_accuracy = accuracy
        if self.best_accuracy is None or accuracy>=self.best_accuracy:
            self.best_accuracy = accuracy
            return True
        else:
            return False
    
    def display(self):
        print("The best accuracy is: {:.2f}%".format(self.best_accuracy*100))
        print("The worst accuracy is: {:.2f}%".format(self.worst_accuracy*100))
        print("The average accuracy is: {:.2f}%".format(self.average_accuracy*100))
        return "The average accuracy is: {:.2f}%".format(self.average_accuracy*100)

def merge_dicts(d1, d2, i):
    for k in d2.keys():
        if k not in d1:
            d1[k] = {}
        for p in d2[k].keys():
            if p not in d1[k]:
                d1[k][p] = d2[k][p]
            else:
                d1[k][p] = (d1[k][p]*i + d2[k][p])/(i+1)
    return d1



def gen_subj_wise_folds(X, Y, subjects, no_folds = 10):
    
    no_subjects = 51
    subjects_per_fold = no_subjects//no_folds
    rng = np.random.default_rng(1)

    idx_list = rng.permutation(no_subjects) + 1
    
    for i in range(no_folds):
        if i==no_folds-1:
            fold_subjs = idx_list[i*subjects_per_fold : ]
        else:
            fold_subjs = idx_list[i*subjects_per_fold : (i+1)*subjects_per_fold]
        mask = np.array([sub in fold_subjs for sub in subjects])
    
        yield X[~mask], Y[~mask], X[mask], Y[mask]


def gen_camera_wise_folds(X, Y, cameras, all_camera_folds = None):
    
    if all_camera_folds is None:
        all_camera_folds = [
            [1], [2], [3], [4], 
            [1, 3], [1, 4], [2, 3], [2, 4], 
            [2, 3, 4], [1, 3, 4], [1, 2, 4], [1, 2, 3]
        ]
    
    for cam in all_camera_folds:
        mask = np.array([c in cam for c in cameras])

        yield X[~mask], Y[~mask], X[mask], Y[mask]
