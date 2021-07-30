import os
import pandas as pd
from numpy import genfromtxt
from utils import merge_dicts
from model import Classifier
import numpy as np

def evaluate_fold(folder_path, X, Y, confusion_path= "confusion_fold.png", method = "random_forest", max_depth = 8, no_trees = 500, lr = 5, predictions = None):
    train_idx = genfromtxt(os.path.join(folder_path, 'train_idx.txt'), delimiter='\n')
    #Y_train = genfromtxt(os.path.join(folder_path, 'y_train.csv'), delimiter=',')
    test_idx = genfromtxt(os.path.join(folder_path, 'test_idx.txt'), delimiter='\n')
    #Y_test = genfromtxt(os.path.join(folder_path, 'y_test.csv'), delimiter=',')
    train_idx = np.asarray([int(i) for i in train_idx])
    test_idx = np.asarray([int(i) for i in test_idx])
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]
    classifier = Classifier(method, max_depth = max_depth, no_estimators = no_trees, lr = lr)

    X_train, X_test = classifier.scale_vectors(X_train, X_test, "scaler.pkl")
    classifier.train(X_train, Y_train)

    # metric_dict = classifier.evaluate(X_test, Y_test, confusion_path = confusion_path)
    #print("The {} classifier with {} decision trees has an accuracy of {}%".format(method, no_trees, 100*accuracy))

    predictions[test_idx] = classifier.model.predict(X_test)

    return predictions

def eval_and_save(X, Y, folders_list, confusion_root, save_path, no_cams = 1):
    predictions = np.asarray([-1 for i in range(len(Y))])
    for i in range(len(folders_list)):
        print("evaluating fold {}".format(i))
        predictions = evaluate_fold(folders_list[i], X, Y, os.path.join(confusion_root, "fold_" + str(i) + ".png"), predictions=predictions)
        df_predictions = pd.DataFrame({"labels": Y, "predictions": predictions})
        print(save_path[:-4] + str(no_cams) + "cam_" + str(i)+".csv")
        df_predictions.to_csv(save_path[:-4] + str(no_cams) + "cam_" + str(i)+".csv", index = False)

    
def main():
    root_path = "../preprocess/camera_wise_fold/"
    data_path = "../preprocess"
    folders_list = [x[0] for x in os.walk(root_path)]
    folders_list = folders_list[1:]
    print(folders_list)
    confusion_root = "confusion_plots"
    os.makedirs(confusion_root, exist_ok=True)

    X = genfromtxt(os.path.join(data_path, 'X_sub_sampled.csv'), delimiter=',')
    Y = genfromtxt(os.path.join(data_path, 'Y_sub_sampled.csv'), delimiter=',')

    # folders_list = ["../preprocess/camera_wise_fold/fold_cam_[1]", "../preprocess/camera_wise_fold/fold_cam_[2]", "../preprocess/camera_wise_fold/fold_cam_[3]", "../preprocess/camera_wise_fold/fold_cam_[4]"]
    # eval_and_save(X, Y, folders_list, confusion_root, "predictions_camera_wise_1cam.csv")  

    folders_list = ["../preprocess/camera_wise_fold/fold_cam_[2, 3]", "../preprocess/camera_wise_fold/fold_cam_[1, 4]", "../preprocess/camera_wise_fold/fold_cam_[2, 4]", "../preprocess/camera_wise_fold/fold_cam_[1, 3]"]
    eval_and_save(X, Y, folders_list, confusion_root, "predictions_camera_wise.csv", 2)

    folders_list = ["../preprocess/camera_wise_fold/fold_cam_[1, 2, 3]", "../preprocess/camera_wise_fold/fold_cam_[1, 2, 4]", "../preprocess/camera_wise_fold/fold_cam_[1, 3, 4]", "../preprocess/camera_wise_fold/fold_cam_[2, 3, 4]"]
    eval_and_save(X, Y, folders_list, confusion_root, "predictions_camera_wise_3cam.csv", 3)  
  
    # df = pd.DataFrame(md)
    # df['Metrics'] = df.index
    # df = df[[df.columns.tolist()[-1]] + df.columns.tolist()[:-1]]
    # df.to_csv("camera_wise_results_10_fold.csv", index = False)

    

        
if __name__ == "__main__":
    main()