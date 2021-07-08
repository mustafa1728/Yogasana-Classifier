import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import argparse as args
from ..utils import create_train_test, save_confusion, get_dataset

def train(num_trees, dataset_path, save_model_path):

    X, Y = get_dataset(dataset_path)

    X_Train, X_Test, Y_Train, Y_Test = create_train_test(X, Y, test_size = 0.25)
    

    # Fitting the classifier into the Training set
    classifier = RandomForestClassifier(n_estimators = num_trees, max_depth=8, criterion = 'gini', random_state = 11)
    classifier.fit(X_Train, Y_Train)
    joblib.dump(classifier, save_model_path)

    accuracy = classifier.score(X_Test, Y_Test)
    print("The random forest with "+str(num_trees)+" decision trees has an accuracy of "+str(100*accuracy)+ "%")

    save_confusion(classifier, X_Test, Y_Test, display_labels = list(class_to_id_mapping.keys()), save_path="confusion_matrix.png")

if __name__ == '__main__':
    ap = args.ArgumentParser(prog='Random Forest Classifier', parents=[get_parent_args()])
    ap.add_argument('--n_trees', '--n', dest='num_trees', nargs=1, type=int, help='Number of Trees in the Classifier')
    ap.parse_args()
    train(num_trees = ap.num_trees, dataset_path = ap.data, save_model_path=ap.save_model_path)
