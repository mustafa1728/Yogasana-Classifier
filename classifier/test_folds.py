from sklearn.preprocessing import StandardScaler
import os
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from numpy import genfromtxt
from ..utils import save_confusion

def test(X_train, Y_train, X_test, Y_test, confusion_path, no_trees):
    sc_X = StandardScaler()
    X_Train = sc_X.fit_transform(X_train)
    X_Test = sc_X.transform(X_test)
    classifier = RandomForestClassifier(n_estimators = no_trees, criterion = 'gini', random_state = 0, max_depth=8)
    classifier.fit(X_Train, Y_train)
    accuracy = classifier.score(X_Test, Y_test)
    print("curent fold has an accuracy of: ", accuracy)
    save_confusion(classifier, X_Test, Y_test, save_path = confusion_path)
    return accuracy

def subject_wise(folder_path, confusion_path= "confusion_fold.png", no_trees = 500):
    X_train = genfromtxt(os.path.join(folder_path, 'x_train.csv'), delimiter=',')
    Y_train = genfromtxt(os.path.join(folder_path, 'y_train.csv'), delimiter=',')
    X_test = genfromtxt(os.path.join(folder_path, 'x_test.csv'), delimiter=',')
    Y_test = genfromtxt(os.path.join(folder_path, 'y_test.csv'), delimiter=',')
    return test(X_train, Y_train, X_test, Y_test, confusion_path, no_trees)
    
def main():
    root_path = "../preprocess/subject_wise_fold/"
    folders_list = [x[0] for x in os.walk(root_path)]
    folders_list = folders_list[1:]
    print(folders_list)
    confusion_root = "confusion_plots"
    os.makedirs(confusion_root, exist_ok=True)
    accuracies = {"path": [], "fold": [], "accuracy": []}

    for i in range(len(folders_list)):
        accuracies["accuracy"].append(subject_wise(folders_list[i], os.path.join(confusion_root, "fold_" + str(i) + ".png")))
        accuracies["path"].append(folders_list[i])
        accuracies["fold"].append(i)
    df = pd.DataFrame(accuracies)
    df.to_csv("subject_wise_results_10_fold.csv", index = False)
        
if __name__ == "__main__":
    main()