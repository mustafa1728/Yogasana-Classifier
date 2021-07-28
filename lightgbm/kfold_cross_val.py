from ..utils import save_confusion, get_dataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os
import joblib
from lightgbm import LGBMClassifier
import pandas as pd

def Kfold_cross_val(n_splits = 10, no_trees = 200, dataset_path = "dataset.csv", save_model_path = "model.z"):
    X, Y = get_dataset(dataset_path)
    
    kf = KFold(n_splits=n_splits, shuffle=True)
    cfms_path = str(n_splits)+"-fold_cfms"
    os.makedirs(cfms_path, exist_ok = True) 
    sc_X = StandardScaler()
    best_accuracy = None
    worst_accuracy = None
    average_accuracy = 0
    count = 0
    fold_id = 0
    k_fold_data = {"fold":[], "train_split_size": [], "test_split_size": [], "accuracy": [], "confusion_plot_path": []}
    for train_index, test_index in kf.split(X):
        fold_id += 1
        X_Train, X_Test = X[train_index], X[test_index]
        Y_Train, Y_Test = Y[train_index], Y[test_index]
        X_Train = sc_X.fit_transform(X_Train)
        X_Test = sc_X.transform(X_Test)
        classifier = LGBMClassifier(boosting_type='goss', max_depth=8, n_estimators = no_trees, random_state = 0, learning_rate=0.5)
        classifier.fit(X_Train, Y_Train)
        accuracy = accuracy_score(Y_Test, classifier.predict(X_Test))
        print("The LightGBM on fold"+str(fold_id)+" with "+str(no_trees)+" estimators has an accuracy of "+str(100*accuracy)+ "%")
        if best_accuracy is None or best_accuracy<=accuracy:
            best_accuracy = accuracy
            joblib.dump(classifier, save_model_path)
        if worst_accuracy is None or worst_accuracy>=accuracy:
            worst_accuracy = accuracy
        average_accuracy += accuracy
        count += 1
        
        confusion_plot_path = os.path.join(cfms_path, "fold_"+str(fold_id))
        save_confusion(classifier, X_Test, Y_Test, confusion_plot_path)

        k_fold_data["fold"].append(fold_id)
        k_fold_data["train_split_size"].append(len(train_index))
        k_fold_data["test_split_size"].append(len(test_index))
        k_fold_data["accuracy"].append(accuracy)
        k_fold_data["confusion_plot_path"].append(confusion_plot_path)
    
    average_accuracy = average_accuracy/count
    save_results_path = str(n_splits)+"-fold cross-validation results.csv"
    df = pd.DataFrame(k_fold_data)
    df.to_csv(save_results_path, index = False)

    print("Best accuracy for "+str(n_splits)+"-fold cross-validation is: ", best_accuracy*100, "%")
    print("Worst accuracy for "+str(n_splits)+"-fold cross-validation is: ", worst_accuracy*100, "%")
    print("Average accuracy for "+str(n_splits)+"-fold cross-validation is: ", average_accuracy*100, "%")

if __name__ == '__main__':
    Kfold_cross_val(n_splits = 10, dataset_path = "../pre-process/dataset.csv", save_model_path = "10-fold_best_model.z")