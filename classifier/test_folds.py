import os
import pandas as pd
from numpy import genfromtxt
from ..utils import AccuracyMeter
from model import Classifier

def evaluate_fold(folder_path, confusion_path= "confusion_fold.png", method = "random_forest", max_depth = 8, no_trees = 500, lr = 5):
    X_train = genfromtxt(os.path.join(folder_path, 'x_train.csv'), delimiter=',')
    Y_train = genfromtxt(os.path.join(folder_path, 'y_train.csv'), delimiter=',')
    X_test = genfromtxt(os.path.join(folder_path, 'x_test.csv'), delimiter=',')
    Y_test = genfromtxt(os.path.join(folder_path, 'y_test.csv'), delimiter=',')

    classifier = Classifier(method, max_depth = max_depth, no_estimators = no_trees, lr = lr)

    X_train, X_test = classifier.scale_vectors(X_train, X_test, "scaler.pkl")
    classifier.train(X_train, Y_train)
    accuracy = classifier.evaluate(X_test, Y_test, confusion_path = confusion_path)
    print("The {} classifier with {} decision trees has an accuracy of {}%".format(method, no_trees, 100*accuracy))

    return accuracy
    
def main():
    root_path = "../preprocess/subject_wise_fold/"
    folders_list = [x[0] for x in os.walk(root_path)]
    folders_list = folders_list[1:]
    print(folders_list)
    confusion_root = "confusion_plots"
    os.makedirs(confusion_root, exist_ok=True)
    accuracies = {"path": [], "fold": [], "accuracy": []}
    accuracy_meter = AccuracyMeter()

    for i in range(len(folders_list)):
        accuracy = evaluate_fold(folders_list[i], os.path.join(confusion_root, "fold_" + str(i) + ".png"))
        accuracies["accuracy"].append(accuracy)
        accuracies["path"].append(folders_list[i])
        accuracies["fold"].append(i)
        accuracy_meter.update(accuracy)
    
    accuracy_meter.display()
    df = pd.DataFrame(accuracies)
    df.to_csv("subject_wise_results_10_fold.csv", index = False)
        
if __name__ == "__main__":
    main()