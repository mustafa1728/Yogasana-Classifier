import numpy as np
import json
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
import joblib
from sklearn.model_selection import StratifiedKFold
import os
import pickle


class_to_id_mapping = {}
id_to_class_mapping = {}

def pre_process_labels(dataset):
    dataset["class"] = dataset["class"].apply(lambda x: still_left_to_still(x))
    dataset["class"] = dataset["class"].apply(lambda x: condition(x))
    classes = list(dataset["class"].unique())
    for i in range(len(classes)):
        class_to_id_mapping[classes[i]] = i
        id_to_class_mapping[i] = classes[i]
    dataset["class"] = dataset["class"].apply(lambda x:class_to_id_mapping[x])
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

def get_dataset(dataset_path):
    dataset = pd.read_csv(dataset_path)
    dataset.dropna(inplace=True)
    indices_to_keep = ~dataset.isin([np.nan, np.inf, -np.inf]).any(1)
    dataset = dataset[indices_to_keep]
    dataset = pre_process_labels(dataset)
    X = dataset.iloc[:, 1:].values
    Y = dataset.iloc[:, 0].values
    X, Y, classes = sub_sample(X, Y, dataset["class"].value_counts().to_dict())
    return X, Y, classes

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

def save_confusion(classifier, X_Test, Y_Test, classes, save_path = "confusion_matrix_sub_sampled.png"):
    fig, ax = plt.subplots(figsize=(20, 16))
    print(classes)
    print(id_to_class_mapping)
    plot_confusion_matrix(
        classifier, 
        X_Test, Y_Test, ax=ax,
        display_labels=[id_to_class_mapping[cls] for cls in classes], 
        cmap=plt.cm.Blues,
        normalize="pred",
        xticks_rotation = "vertical"
    )
    plt.savefig(save_path, dpi = 300)

def train(no_trees = 200, max_depth = 8, dataset_path = "dataset.csv", save_model_path = "model.z", save_mapping = "ids_to_class.json"):

    X, Y, classes = get_dataset(dataset_path)

    # Splitting the dataset into the Training set and Test set
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.25, random_state = 0, stratify=Y)

    # Feature Scaling
    sc_X = StandardScaler()
    X_Train = sc_X.fit_transform(X_Train)
    X_Test = sc_X.transform(X_Test)

    with open("scalar.pkl", "wb") as f:
        pickle.dump(sc_X, f)

    # Fitting the classifier into the Training set
    classifier = RandomForestClassifier(n_estimators = no_trees, criterion = 'gini', random_state = 11, max_depth=max_depth)
    classifier.fit(X_Train,Y_Train)
    joblib.dump(classifier, save_model_path)

    accuracy = classifier.score(X_Test, Y_Test)
    print("The random forest with "+str(no_trees)+" decision trees has an accuracy of "+str(100*accuracy)+ "%")
    
    save_confusion(classifier, X_Test, Y_Test, classes)
    
    with open(save_mapping, 'w') as f:
        json.dump(id_to_class_mapping, f)

def load_model(model_weights_path):
    classifier = joblib.load(model_weights_path)
    return classifier

def predict_class(input_features, model_weights_path = "model.z", saved_mapping = "ids_to_class.json"):
    model = load_model(model_weights_path)
    class_pred = model.predict([input_features])
    with open(saved_mapping) as f:
        id_to_class_mapping = json.load(f)
    return id_to_class_mapping[class_pred]

def Kfold_cross_val(n_splits = 10, no_trees = 200, max_depth = 8, dataset_path = "dataset.csv", save_model_path = "model.z"):
    X, Y, classes = get_dataset(dataset_path)
    
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2)
    cfms_path = str(n_splits)+"-fold_cfms_subsampled"
    os.makedirs(cfms_path, exist_ok = True) 
    sc_X = StandardScaler()
    best_accuracy = None
    worst_accuracy = None
    average_accuracy = 0
    fold_id = 0
    k_fold_data = {"fold":[], "train_split_size": [], "test_split_size": [], "accuracy": [], "confusion_plot_path": []}
    for train_index, test_index in kf.split(X, Y):
        fold_id += 1
        X_Train, X_Test = X[train_index], X[test_index]
        Y_Train, Y_Test = Y[train_index], Y[test_index]
        X_Train = sc_X.fit_transform(X_Train)
        X_Test = sc_X.transform(X_Test)
        classifier = RandomForestClassifier(n_estimators = no_trees, max_depth=max_depth, criterion = 'gini', random_state = 11)
        classifier.fit(X_Train, Y_Train)
        accuracy = classifier.score(X_Test, Y_Test)
        print("The random forest on fold"+str(fold_id)+" with "+str(no_trees)+" decision trees has an accuracy of "+str(100*accuracy)+ "%")
        if best_accuracy is None or best_accuracy<=accuracy:
            best_accuracy = accuracy
            joblib.dump(classifier, save_model_path)
        if worst_accuracy is None or worst_accuracy>=accuracy:
            worst_accuracy = accuracy
        average_accuracy += accuracy
        
        confusion_plot_path = os.path.join(cfms_path, "fold_"+str(fold_id))
        save_confusion(classifier, X_Test, Y_Test, classes, confusion_plot_path)

        k_fold_data["fold"].append(fold_id)
        k_fold_data["train_split_size"].append(len(train_index))
        k_fold_data["test_split_size"].append(len(test_index))
        k_fold_data["accuracy"].append(accuracy)
        k_fold_data["confusion_plot_path"].append(confusion_plot_path)
    

    save_results_path = str(n_splits)+"-fold_cross-validation_results_subsampled.csv"
    df = pd.DataFrame(k_fold_data)
    df.to_csv(save_results_path, index = False)

    average_accuracy = average_accuracy / n_splits
    print("Best accuracy for "+str(n_splits)+"-fold cross-validation is: ", best_accuracy*100, "%")
    print("Worst accuracy for "+str(n_splits)+"-fold cross-validation is: ", worst_accuracy*100, "%")
    print("Average accuracy for "+str(n_splits)+"-fold cross-validation is: ", average_accuracy*100, "%")

    k_fold_data["Average Accuracy"] = ['{0:.2f}%'.format(average_accuracy*100)]
    k_fold_data["Worst Accuracy"] = ['{0:.2f}%'.format(worst_accuracy*100)]
    k_fold_data["Best Accuracy"] = ['{0:.2f}%'.format(best_accuracy*100)]
    df = pd.DataFrame(k_fold_data)
    df.to_csv(save_results_path, index = False)


# train(no_trees = 500, max_depth = 6, dataset_path = "../preprocess/dataset.csv", save_model_path = "model_subsampled.z")
Kfold_cross_val(n_splits = 10, no_trees=500, max_depth = 6, dataset_path = "../preprocess/dataset.csv", save_model_path = "10-fold_best_model_subsampled.z")
