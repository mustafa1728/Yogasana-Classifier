import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
import joblib
import argparse as args
from ..utils import create_train_test, get_parent_args, pre_process_labels

def train(num_trees, dataset_path, save_model_path):

    dataset = pd.read_csv(dataset_path)
    dataset.dropna(inplace=True)
    dataset, class_to_id_mapping, id_to_class_mapping = pre_process_labels(dataset)
    X = dataset.iloc[:, 1:].values
    Y = dataset.iloc[:, 0].values

    
    X_Train, X_Test, Y_Train, Y_Test = create_train_test(X, Y, test_size = 0.25)
    

    # Fitting the classifier into the Training set
    classifier = RandomForestClassifier(n_estimators = num_trees, max_depth=8, criterion = 'gini', random_state = 11)
    classifier.fit(X_Train, Y_Train)
    joblib.dump(classifier, save_model_path)

    accuracy = classifier.score(X_Test, Y_Test)
    print("The random forest with "+str(num_trees)+" decision trees has an accuracy of "+str(100*accuracy)+ "%")

    plot_confusion_matrix(
        classifier, 
        X_Test, Y_Test, 
        display_labels=list(class_to_id_mapping.keys()), 
        cmap=plt.cm.Blues,
        normalize="pred"
    )
    plt.savefig("confusion_matrix.png", pad_inches = 2, dpi = 300)

if __name__ == '__main__':
    ap = args.ArgumentParser(prog='Random Forest Classifier', parents=[get_parent_args()])
    ap.add_argument('--n_trees', '--n', dest='num_trees', nargs=1, type=int, help='Number of Trees in the Classifier')
    ap.parse_args()
    train(num_trees = ap.num_trees, dataset_path = ap.data, save_model_path=ap.save_model_path)
