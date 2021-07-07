import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import joblib

class_to_id_mapping = {}
id_to_class_mapping = {}

def pre_process_labels(dataset):
    classes = list(dataset["class"].unique())
    for i in range(len(classes)):
        class_to_id_mapping[classes[i]] = i
        id_to_class_mapping[i] = classes[i]
    dataset["class"] = dataset["class"].apply(lambda x:class_to_id_mapping[x])
    return dataset

def train(no_trees = 200, dataset_path = "dataset.csv", save_model_path = "model.z", save_mapping = "ids_to_class.json"):

    dataset = pd.read_csv(dataset_path)
    dataset = pre_process_labels(dataset)
    X = dataset.iloc[:, 1:].values
    Y = dataset.iloc[:, 0].values

    # Splitting the dataset into the Training set and Test set
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

    # Feature Scaling
    sc_X = StandardScaler()
    X_Train = sc_X.fit_transform(X_Train)
    X_Test = sc_X.transform(X_Test)

    # Fitting the classifier into the Training set
    classifier = RandomForestClassifier(n_estimators = no_trees, criterion = 'entropy', random_state = 0)
    classifier.fit(X_Train,Y_Train)
    joblib.dump(classifier, save_model_path)

    accuracy = classifier.score(X_Test, Y_Test)
    print("The random forest with "+str(no_trees)+" decision trees has an accuracy of "+str(100*accuracy)+ "%")

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
    

    with open(save_mapping, 'w') as f:
        json.dump(id_to_class_mapping, f)

def load_model(model_weights_path):
    classifier = joblib.load(model_weights_path)
    return classifier

def predict_class(input_features, model_weights_path = "model.z", saved_mapping = "ids_to_class.json"):
    model = load_model(model_weights_path)
    class_pred = model.predict([input_features])
    with open(saved_mapping) as f:
        id_to_class_mapping = json.load(f)
    return id_to_class_mapping[class_pred]

train(no_trees = 200, dataset_path = "../pre-process/dataset.csv")
# train(no_trees = 200)
