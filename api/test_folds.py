import os
import pandas as pd
from numpy import genfromtxt
from utils import merge_dicts, gen
from classifier.model import Classifier

def evaluate_fold(X_train, Y_train, X_test, Y_test, confusion_path= "confusion_fold.png", method = "random_forest", max_depth = 8, no_trees = 500, lr = 5):
    #train_idx = genfromtxt(os.path.join(folder_path, 'train_idx.txt'), delimiter='\n')
    #Y_train = genfromtxt(os.path.join(folder_path, 'y_train.csv'), delimiter=',')
    #test_idx = genfromtxt(os.path.join(folder_path, 'test_idx.txt'), delimiter='\n')
    #Y_test = genfromtxt(os.path.join(folder_path, 'y_test.csv'), delimiter=',')
    
    classifier = Classifier(method, max_depth = max_depth, no_estimators = no_trees, lr = lr)

    X_train, X_test = classifier.scale_vectors(X_train, X_test, "scaler.pkl")
    classifier.train(X_train, Y_train)

    metric_dict = classifier.evaluate(X_test, Y_test, confusion_path = confusion_path)
    #print("The {} classifier with {} decision trees has an accuracy of {}%".format(method, no_trees, 100*accuracy))

    predictions = classifier.model.predict(X_test).tolist()

    return metric_dict, predictions
    
def main():
    #root_path = "../preprocess/subject_wise_fold/"
    #data_path = "../preprocess"
    #folders_list = [x[0] for x in os.walk(root_path)]
    #folders_list = folders_list[1:]
    #print(folders_list)
    confusion_root = "confusion_plots"
    os.makedirs(confusion_root, exist_ok=True)

    #X = genfromtxt(os.join.path(data_path, 'X_sub_sampled.csv'), delimiter=',')
    #Y = genfromtxt(os.join.path(data_path, 'Y_sub_sampled.csv'), delimiter=',')

    X_train_list, Y_train_list, X_test_list, Y_test_list = gen('../preprocess/dataset.csv')

    pred = []
    lab = []
    md = {}

    for i, (X_train, Y_train, X_test, Y_test) in enumerate(zip(X_train_list, Y_train_list, X_test_list, Y_test_list)):
        lab += Y_test.tolist()
        metrtic_dict, predictions = evaluate_fold(X_train, Y_train, X_test, Y_test, os.path.join(confusion_root, "fold_" + str(i) + ".png"))
        pred += predictions
        md = merge_dicts(md, metrtic_dict, i)
    
    #df = pd.DataFrame(md)
    #df['Metrics'] = df.index
    #df = df[[df.columns.tolist()[-1]] + df.columns.tolist()[:-1]]
    #df.to_csv("camera_wise_results_10_fold.csv", index = False)

    df_predictions = pd.DataFrame({"labels": lab, "predictions": pred})
    df_predictions.to_csv("predictions_cam_wise.csv", index = False)

        
if __name__ == "__main__":
    main()