import os
import pandas as pd
from numpy import genfromtxt
from utils import merge_dicts
from model import Classifier

def evaluate_fold(folder_path, X, Y, confusion_path= "confusion_fold.png", method = "random_forest", max_depth = 8, no_trees = 500, lr = 5, predictions = None):
    train_idx = genfromtxt(os.path.join(folder_path, 'train_idx.txt'), delimiter='\n')
    #Y_train = genfromtxt(os.path.join(folder_path, 'y_train.csv'), delimiter=',')
    test_idx = genfromtxt(os.path.join(folder_path, 'test_idx.txt'), delimiter='\n')
    #Y_test = genfromtxt(os.path.join(folder_path, 'y_test.csv'), delimiter=',')
    
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]
    classifier = Classifier(method, max_depth = max_depth, no_estimators = no_trees, lr = lr)

    X_train, X_test = classifier.scale_vectors(X_train, X_test, "scaler.pkl")
    classifier.train(X_train, Y_train)

    metric_dict = classifier.evaluate(X_test, Y_test, confusion_path = confusion_path)
    #print("The {} classifier with {} decision trees has an accuracy of {}%".format(method, no_trees, 100*accuracy))

    predictions[test_idx] = classifier.model.predict(X_test)

    return metric_dict, predictions
    
def main():
    root_path = "../preprocess/subject_wise_fold/"
    data_path = "../preprocess"
    folders_list = [x[0] for x in os.walk(root_path)]
    folders_list = folders_list[1:]
    print(folders_list)
    confusion_root = "confusion_plots"
    os.makedirs(confusion_root, exist_ok=True)

    X = genfromtxt(os.join.path(data_path, 'X_sub_sampled.csv'), delimiter=',')
    Y = genfromtxt(os.join.path(data_path, 'Y_sub_sampled.csv'), delimiter=',')

    predictions = [-1 for i in range(len(Y))]
    md = {}

    for i in range(len(folders_list)):
        metrtic_dict, predictions = evaluate_fold(folders_list[i], X, Y, os.path.join(confusion_root, "fold_" + str(i) + ".png", predictions=predictions))
        md = merge_dicts(md, metrtic_dict, i)
    
    df = pd.DataFrame(md)
    df['Metrics'] = df.index
    df = df[[df.columns.tolist()[-1]] + df.columns.tolist()[:-1]]
    df.to_csv("subject_wise_results_10_fold.csv", index = False)

    df_predictions = pd.DataFrame({"labels": Y, "predictions": predictions})
    df_predictions.to_csv("predictions_subj_wise.csv", index = False)

        
if __name__ == "__main__":
    main()