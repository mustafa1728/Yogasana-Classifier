import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
import joblib
from sklearn.model_selection import StratifiedKFold
import os
import pickle

from ..utils import get_dataset, save_confusion

def train(no_trees = 200, max_depth = 8, dataset_path = "dataset.csv", save_model_path = "model.z", model = "adaboost"):

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
    if model == "adaboost":
        classifier = AdaBoostClassifier(DecisionTreeClassifier(criterion='gini', max_depth=max_depth), n_estimators = no_trees, random_state = 0)
    elif model == "random_forest":
        classifier = RandomForestClassifier(n_estimators = no_trees, criterion = 'entropy', random_state = 0, max_depth=max_depth)
    elif model == "LGBM":
        classifier = LGBMClassifier(boosting_type='goss', max_depth=max_depth, n_estimators = no_trees, random_state = 0, learning_rate=0.5)
    classifier.fit(X_Train,Y_Train)
    joblib.dump(classifier, save_model_path)
    if model == "LGBM":
        accuracy = accuracy_score(Y_Test, classifier.predict(X_Test))
    else:
        accuracy = classifier.score(X_Test, Y_Test)
    print("The {} classifier with {} decision trees has an accuracy of {}%".format(model, no_trees, 100*accuracy))
    
    save_confusion(classifier, X_Test, Y_Test, classes)

def Kfold_cross_val(n_splits = 10, no_trees = 200, max_depth = 8, dataset_path = "dataset.csv", save_model_path = "model.z", model = "adaboost"):
    X, Y, classes = get_dataset(dataset_path)
    
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2)
    cfms_path = "{}-fold_cfms_subsampled_depth_{}".format(n_splits, max_depth)
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
        if model == "adaboost":
            classifier = AdaBoostClassifier(DecisionTreeClassifier(criterion='gini', max_depth=max_depth), n_estimators = no_trees, random_state = 0)
        elif model == "random_forest":
            classifier = RandomForestClassifier(n_estimators = no_trees, criterion = 'entropy', random_state = 0, max_depth = max_depth)
        elif model == "LGBM":
            classifier = LGBMClassifier(boosting_type='goss', max_depth=max_depth, n_estimators = no_trees, random_state = 0, learning_rate=0.5)
        classifier.fit(X_Train, Y_Train)
        if model == "LGBM":
            accuracy = accuracy_score(Y_Test, classifier.predict(X_Test))
        else:
            accuracy = classifier.score(X_Test, Y_Test)
        print("The {} classifier with {} decision trees has an accuracy of {}%".format(model, no_trees, 100*accuracy))
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
    
    save_results_path = "{}_{}-fold_cross-validation_results_max_depth_{}.csv".format(model, n_splits, max_depth)
    df = pd.DataFrame(k_fold_data)
    df.to_csv(save_results_path, index = False)

    average_accuracy = average_accuracy / n_splits
    print("Best accuracy for "+str(n_splits)+"-fold cross-validation is: ", best_accuracy*100, "%")
    print("Worst accuracy for "+str(n_splits)+"-fold cross-validation is: ", worst_accuracy*100, "%")
    print("Average accuracy for "+str(n_splits)+"-fold cross-validation is: ", average_accuracy*100, "%")

    # k_fold_data["Average Accuracy"] = ['{0:.2f}%'.format(average_accuracy*100)]
    # k_fold_data["Worst Accuracy"] = ['{0:.2f}%'.format(worst_accuracy*100)]
    # k_fold_data["Best Accuracy"] = ['{0:.2f}%'.format(best_accuracy*100)]
    df = pd.DataFrame(k_fold_data)
    df.to_csv(save_results_path, index = False)

# for i in range(9, 15):
#     print("value of depth: ", i)
#     train(no_trees = 500, max_depth = i, dataset_path = "../../dataset.csv", save_model_path = "model_subsampled_temp.z")
Kfold_cross_val(n_splits = 10, no_trees=500, max_depth = 8, dataset_path = "../../dataset.csv", save_model_path = "10-fold_best_model_subsampled.z")
