import os
import json
import pandas as pd
import logging
import numpy as np

def gen_camera_wise_folds(cameras, X, Y):
    
    all_cameras = [1, 2, 3, 4]
    fold_root_dir_path = "camera_wise_fold"
    fold_dir_templ = "fold_cam_{}"
    os.makedirs(fold_root_dir_path, exist_ok=True)


    for cam in all_cameras:
        X_train = X[cameras==cam]
        Y_train = Y[cameras==cam]
        X_test = X[cameras==cam]
        Y_test = Y[cameras==cam]

        fold_path = os.path.join(fold_root_dir_path, fold_dir_templ.format(cam))
        os.makedirs(fold_path, exist_ok=True)

        np.savetxt(os.path.join(fold_path, 'x_train.csv'), X_train, delimiter=',')
        np.savetxt(os.path.join(fold_path, 'y_train.csv'), Y_train, delimiter=',')
        np.savetxt(os.path.join(fold_path, 'x_test.csv'), X_test, delimiter=',')
        np.savetxt(os.path.join(fold_path, 'y_test.csv'), Y_test, delimiter=',')

    


        

