import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import joblib
from ..utils import create_train_test, pre_process_labels

def train(num_trees = 200, dataset_path = "dataset.csv", save_model_path = "model.z"):

    dataset = pd.read_csv(dataset_path)
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
    # predictions = classifier.predict(X_Test)
    # cf_matrix = confusion_matrix(Y_Test, predictions)
    # plt.imshow(cf_matrix, cmap='viridis')
    # plt.colorbar()
    # plt.savefig("confusion_matrix.png", pad_inches = 1, dpi = 300)
    # cf_matrix_plot = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')
    # cf_matrix_plot.get_figure().savefig("confusion_matrix.png", pad_inches = 1, dpi = 300)
    

    #with open(save_mapping, 'w') as f:
    #    json.dump(id_to_class_mapping, f)

if __name__ == '__main__':
    train(num_trees = 200, dataset_path = "../pre-process/dataset.csv")
