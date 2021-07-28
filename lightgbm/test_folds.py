from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import os
import joblib
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from numpy import genfromtxt
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import plot_confusion_matrix

def save_confusion(classifier, X_Test, Y_Test, display_labels=None, save_path = "confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(15, 12))
    plot_confusion_matrix(
        classifier, 
        X_Test, Y_Test, ax=ax,
        display_labels=display_labels, 
        cmap=plt.cm.Blues,
        normalize="pred",
        xticks_rotation = "vertical"
    )
    plt.savefig(save_path, dpi = 300)


def test(X_train, Y_train, X_test, Y_test, confusion_path, no_trees):
    sc_X = StandardScaler()
    X_Train = sc_X.fit_transform(X_train)
    X_Test = sc_X.transform(X_test)
    classifier = LGBMClassifier(boosting_type='goss', max_depth=8, n_estimators = no_trees, random_state = 0, learning_rate=0.5)
    classifier.fit(X_Train, Y_train)
    accuracy = accuracy_score(Y_test, classifier.predict(X_Test))
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
    df.to_csv("subject_wise_results.csv", index = False)
        
if __name__ == "__main__":
    main()