from math import log
from sklearn.model_selection import StratifiedKFold
from utils import merge_dicts
import os
import pandas as pd
import numpy as np

from classifier.model import Classifier
from utils import get_dataset, gen_subj_wise_folds, gen_camera_wise_folds, AccuracyMeter
import logging

def Kfold_cross_val(
    max_depth = 8, 
    no_trees = 500, 
    n_splits = 10, 
    dataset_path = "dataset.csv", 
    save_model_path = "model.z", 
    predictions_path=None, 
    method = "random_forest", 
    lr = 5, 
    split_type="frame",
    log_path="eval.log"
):

    logging.basicConfig(filename=log_path, filemode='a', format='%(levelname)s | %(message)s', level=logging.INFO)
    
    X, Y, subjects, cams = get_dataset(dataset_path, include_as_ratio=True, return_subj=True, return_cams=True)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2)

    cfms_path = "{}-fold_cfms_subsampled_depth_{}".format(n_splits, max_depth)
    os.makedirs(cfms_path, exist_ok = True) 
    accuracy_meter = AccuracyMeter()
    fold_id = 0
    predictions = []
    labels = []

    if split_type == "frame":
        generator = kf.split(X, Y)
    elif split_type == "subject":
        generator = gen_subj_wise_folds(X, Y, subjects, n_splits)
    elif split_type == "camera":
        generator = gen_camera_wise_folds(X, Y, cams)
    else:
        raise ValueError("split_type should be one of [frame, subject, camera]. Received {}".format(split_type))

    for train_index, test_index in generator:
        fold_id += 1
        confusion_plot_path = os.path.join(cfms_path, "fold_"+str(fold_id))

        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        classifier = Classifier(method, max_depth = max_depth, no_estimators = no_trees, lr = lr)

        X_train, X_test = classifier.scale_vectors(X_train, X_test, "scaler.pkl")
        classifier.train(X_train, Y_train)
        accuracy = classifier.evaluate(X_test, Y_test, confusion_path = confusion_plot_path)
        logging.info("Fold {:2d} - Accuracy: {:.2f}%".format(fold_id, 100*accuracy))

        predictions.append(classifier.model.predict(X_test))
        labels.append(Y_test)

        is_best = accuracy_meter.update(accuracy)
        is_best = True
        if is_best:
           classifier.save_model(save_model_path)

  
    logging.log(accuracy_meter.display())

    if predictions_path is not None:
        df_predictions = pd.DataFrame({"labels": np.concatenate(labels), "predictions": np.concatenate(predictions)})
        df_predictions.to_csv(predictions_path, index = False)

