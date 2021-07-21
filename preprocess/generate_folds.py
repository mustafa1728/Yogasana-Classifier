import os
import json
import pandas as pd
import logging
import numpy as np

def gen_camera_wise_folds(X, Y, cameras):
    
    all_cameras = ["1", "2", "3", "4"]
    fold_root_dir_path = "camera_wise_fold"
    fold_dir_templ = "fold_cam_{}"
    os.makedirs(fold_root_dir_path, exist_ok=True)

    

    for cam in all_cameras:
        print(cameras[:5], cam)
        X_train = X[cameras==cam]
        Y_train = Y[cameras==cam]
        X_test = X[cameras!=cam]
        Y_test = Y[cameras!=cam]

        print(X.shape, X_train.shape)
        print(Y.shape, Y_train.shape)

        fold_path = os.path.join(fold_root_dir_path, fold_dir_templ.format(cam))
        os.makedirs(fold_path, exist_ok=True)

        np.savetxt(os.path.join(fold_path, 'x_train.csv'), X_train, delimiter=',')
        np.savetxt(os.path.join(fold_path, 'y_train.csv'), Y_train, delimiter=',')
        np.savetxt(os.path.join(fold_path, 'x_test.csv'), X_test, delimiter=',')
        np.savetxt(os.path.join(fold_path, 'y_test.csv'), Y_test, delimiter=',')


def gen_subj_wise_folds(X, Y, subjects, no_folds = 10, subjects_per_fold = 5):
    
    no__subjects = 52
    fold_root_dir_path = "subject_wise_fold"
    fold_dir_templ = "fold_subj_{}"
    os.makedirs(fold_root_dir_path, exist_ok=True)
    rng = np.random.default_rng(1)

    for i in range(no_folds):
        fold_subjs = rng.choice(no__subjects, size = subjects_per_fold, replace = False)
        cond = False
        for sub in fold_subjs:
            cond = cond | (subjects == sub)

        X_train = X[~cond]
        Y_train = Y[~cond]
        X_test = X[cond]
        Y_test = Y[cond]

        subjs_string = "subjs"
        for sub in fold_subjs:
            subjs_string += "_" + str(sub)
        fold_path = os.path.join(fold_root_dir_path, fold_dir_templ.format(subjs_string))
        os.makedirs(fold_path, exist_ok=True)

        np.savetxt(os.path.join(fold_path, 'x_train.csv'), X_train, delimiter=',')
        np.savetxt(os.path.join(fold_path, 'y_train.csv'), Y_train, delimiter=',')
        np.savetxt(os.path.join(fold_path, 'x_test.csv'), X_test, delimiter=',')
        np.savetxt(os.path.join(fold_path, 'y_test.csv'), Y_test, delimiter=',')

# can be imported
def still_left_to_still(x):
    if x == "Still_left": 
        return "Still" 
    else:  
        return x

# can be imported
def condition(x):
    if x == "None": 
        return "Still" 
    else:  
        return x

# can be imported
def pre_process_labels(dataset):
    class_to_id_mapping = {}
    id_to_class_mapping = {}
    dataset["class"] = dataset["class"].apply(lambda x: condition(x))
    dataset["class"] = dataset["class"].apply(lambda x: still_left_to_still(x))
    classes = list(dataset["class"].unique())
    for i in range(len(classes)):
        class_to_id_mapping[classes[i]] = i
        id_to_class_mapping[i] = classes[i]
    dataset["class"] = dataset["class"].apply(lambda x:class_to_id_mapping[x])
    with open("ids_to_class_curr.json", 'w') as f:
        json.dump(id_to_class_mapping, f)
    return dataset

# can be imported
def sub_sample(X, Y, class_counts):
    no_samples_per_class = 6000
    classes = [cls for cls in list(class_counts.keys())[:12] ]

    rng = np.random.default_rng(1)
    X_subset_list = []
    Y_subset_list = []

    for cls in classes:
        total_samples = X[Y==cls]
        total_labels = Y[Y==cls]
        idx = rng.choice(total_samples.shape[0], size = no_samples_per_class, replace = False)
        
        X_subset_list.append(total_samples[idx])
        Y_subset_list.append(total_labels[idx])

    X_subset = np.concatenate(X_subset_list, axis = 0)
    Y_subset = np.concatenate(Y_subset_list, axis = 0)
    return X_subset, Y_subset, classes

def main(dataset_path):
    dataset = pd.read_csv(dataset_path)
    dataset.dropna(inplace=True)
    indices_to_keep = ~dataset.isin([np.nan, np.inf, -np.inf]).any(1)
    dataset = dataset[indices_to_keep]
    dataset = pre_process_labels(dataset)
    X = dataset.iloc[:, 4:].values
    Y = dataset.iloc[:, 3].values
    # X, Y, classes = sub_sample(X, Y, dataset["class"].value_counts().to_dict())
    
    # gen_camera_wise_folds(X, Y, cameras=dataset.iloc[:, 0].apply(lambda x: int(x)).values)
    gen_subj_wise_folds(X, Y, subjects=dataset.iloc[:, 1].apply(lambda x: int(x[-3:])).values)

if __name__ == '__main__':
    main("../../dataset.csv")