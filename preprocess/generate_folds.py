import os
import json
import pandas as pd
import logging
import numpy as np

def gen_camera_wise_folds(X, Y, cameras):
    
    all_cameras = [1, 2, 3, 4]
    fold_root_dir_path = "camera_wise_fold"
    fold_dir_templ = "fold_cam_{}"
    os.makedirs(fold_root_dir_path, exist_ok=True)


    for cam in all_cameras:
        X_train = X[cameras==cam]
        Y_train = Y[cameras==cam]
        X_test = X[cameras!=cam]
        Y_test = Y[cameras!=cam]

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

        X_train = X[cond]
        Y_train = Y[cond]
        X_test = X[~cond]
        Y_test = Y[~cond]

        subjs_string = "subjs"
        for sub in fold_subjs:
            subjs_string += "_" + str(sub)
        fold_path = os.path.join(fold_root_dir_path, fold_dir_templ.format(subjs_string))
        os.makedirs(fold_path, exist_ok=True)

        np.savetxt(os.path.join(fold_path, 'x_train.csv'), X_train, delimiter=',')
        np.savetxt(os.path.join(fold_path, 'y_train.csv'), Y_train, delimiter=',')
        np.savetxt(os.path.join(fold_path, 'x_test.csv'), X_test, delimiter=',')
        np.savetxt(os.path.join(fold_path, 'y_test.csv'), Y_test, delimiter=',')
