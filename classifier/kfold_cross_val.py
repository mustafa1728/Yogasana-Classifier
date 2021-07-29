from sklearn.model_selection import StratifiedKFold
from utils import merge_dicts
import os
import pandas as pd

from model import Classifier
from utils import get_dataset, AccuracyMeter

def Kfold_cross_val(n_splits = 10, no_trees = 500, max_depth = 8, dataset_path = "dataset.csv", save_model_path = "model.z", method = "random_forest", lr = 5):
    X, Y, _ = get_dataset(dataset_path)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2)

    cfms_path = "{}-fold_cfms_subsampled_depth_{}".format(n_splits, max_depth)
    os.makedirs(cfms_path, exist_ok = True) 
    accuracy_meter = AccuracyMeter()
    fold_id = 0
    #k_fold_data = {"fold":[], "train_split_size": [], "test_split_size": [], "accuracy": [], "confusion_plot_path": []}
    md = {}
    for train_index, test_index in kf.split(X, Y):
        fold_id += 1
        confusion_plot_path = os.path.join(cfms_path, "fold_"+str(fold_id))

        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        classifier = Classifier(method, max_depth = max_depth, no_estimators = no_trees, lr = lr)

        X_train, X_test = classifier.scale_vectors(X_train, X_test, "scaler.pkl")
        classifier.train(X_train, Y_train)
        metric_dict = classifier.evaluate(X_test, Y_test, confusion_path = confusion_plot_path)

        md = merge_dicts(md, metric_dict)
        #print("The {} classifier with {} decision trees has an accuracy of {}%".format(method, no_trees, 100*accuracy))

        #is_best = accuracy_meter.update(accuracy)
        #if is_best:
        #    classifier.save_model(save_model_path)

        #k_fold_data["fold"].append(fold_id)
        #k_fold_data["train_split_size"].append(len(train_index))
        #k_fold_data["test_split_size"].append(len(test_index))
        #k_fold_data["accuracy"].append(accuracy)
        #k_fold_data["confusion_plot_path"].append(confusion_plot_path)
    
    #accuracy_meter.display()

    save_results_path = "{}_{}-fold_cross-validation_results_max_depth_{}.csv".format(method, n_splits, max_depth)
    df = pd.DataFrame(md)
    df['Metrics'] = df.index
    df = df[[df.columns.tolist()[-1]] + df.columns.tolist()[:-1]]
    df.to_csv(save_results_path, index = False)

if __name__ == '__main__':
    Kfold_cross_val(max_depth = 20, n_splits = 10, dataset_path = "/Users/mustafa/Desktop/IIT Delhi/Internship stuff/SURA/dataset.csv", save_model_path = "10-fold_best_model.z")
