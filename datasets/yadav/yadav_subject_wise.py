from sklearn.model_selection import StratifiedKFold
from utils import merge_dicts
import os
import pandas as pd

from classifier.model import Classifier
from utils import get_dataset, AccuracyMeter
import numpy as np

def subject_wise(no_trees = 500, max_depth = 8, dataset_path = "dataset.csv", save_model_path = "model.z", method = "random_forest", lr = 5, cfms_path = "confusion"):
    X, Y = get_dataset(dataset_path)
    print(X.shape)
    os.makedirs(cfms_path, exist_ok=True)
    predictions = np.asarray([-1 for i in range(len(Y))])
    # print(Y[0])
    dataset = pd.read_csv(dataset_path)
    subjects = dataset["subject"]
    subjects_list = subjects.unique()
    classes = dataset["class"].unique()
    accuracy_meter = AccuracyMeter()
    rng = np.random.default_rng(1)
    print(len(subjects), len(classes))
    idx_list = rng.permutation(len(subjects_list)) 
    # print(idx_list)
    no_folds = 5
    for i in range(no_folds):
        if i == no_folds - 1:
            current_test_idx = idx_list[3*i:]
        else:
            current_test_idx = idx_list[3*i:3*(i+1)]
        current_subjs = [subjects_list[k] for k in current_test_idx]

        train_index = [k for k in range(len(subjects)) if not subjects[k] in current_subjs]
        test_index = [k for k in range(len(subjects)) if subjects[k] in current_subjs]
        X_train = X[train_index]
        Y_train = Y[train_index]
        X_test = X[test_index]
        Y_test = Y[test_index]

        print("train: ", len(Y_train), "test: ", len(Y_test))
        
        classifier = Classifier(method, max_depth = max_depth, no_estimators = no_trees, lr = lr)


        X_train, X_test = classifier.scale_vectors(X_train, X_test, "scaler.pkl")
        classifier.train(X_train, Y_train)
        confusion_plot_path = os.path.join(cfms_path, "fold_"+str(i))
        accuracy = classifier.evaluate(X_test, Y_test, confusion_path = confusion_plot_path)

        predictions[test_index] = classifier.model.predict(X_test)
        print("The {} classifier has an accuracy of {}%".format(method, 100*accuracy))

        is_best = accuracy_meter.update(accuracy)
        if is_best:
           classifier.save_model(save_model_path)

        df_predictions = pd.DataFrame({"labels": Y, "predictions": predictions})
        df_predictions.to_csv("yadav_predictions_subject_wise_no_curation.csv", index = False)
   

if __name__ == '__main__':
    subject_wise(max_depth = None, dataset_path = "/Users/mustafa/Desktop/yoga/Yogasana-Classifier/preprocess/yadav_normalised_no_curation_dataset.csv", save_model_path = "yadav_best_model.z", cfms_path = "yadav_subject_wise")

    
