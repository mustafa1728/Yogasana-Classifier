import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
# from lightgbm import LGBMClassifier
import joblib
import pickle

from utils import get_dataset, save_confusion

class Classifier():
    '''
        Generic classifier 
        Currently supported methods:
            adaboost
            random_forest
            LGBM
        Parameters: 
            max_depth       : maximum depth of any tree
            no_estimators   : number of trees in the ensemble
            lr              : the learning rate required by AdaBoost and LGBM
            random_state    : seed for random number generation. Used to make code reproducable
    '''
    def __init__(self, method, max_depth = 8, no_estimators = 500, lr = 0.5, random_state = 0):
        self.method = method
        self.max_depth = max_depth
        self.no_estimators = no_estimators
        self.random_state = random_state
        self.lr = lr
        self.model = None
        self.sc_X = None

    def train(self, X_train, Y_train):
        if self.method == "adaboost":
            self.model = AdaBoostClassifier(DecisionTreeClassifier(criterion='gini', max_depth=self.max_depth), n_estimators = self.no_estimators, random_state = self.random_state, learning_rate=self.lr)
        elif self.method == "random_forest":
            self.model = RandomForestClassifier(n_estimators = self.no_estimators, criterion = 'entropy', random_state = self.random_state, max_depth=self.max_depth)
        elif self.method == "LGBM":
            self.model = LGBMClassifier(boosting_type='goss', max_depth=self.max_depth, n_estimators = self.no_estimators, random_state = self.random_state, learning_rate=self.lr)
        self.model.fit(X_train,Y_train)
        return self.model

    def save_model(self, save_path):
        joblib.dump(self.model, save_path)

    def load_model(self, model_path):
        self.model = joblib.load(model_path)
        return self.model

    def evaluate(self, X_test, Y_test, model = None, confusion_path = None):
        if model is None:
            model = self.model
        metric_dict = {i:{'acc': 0, 'prec': 0, 'rec': 0, 'f1': 0} for i in model.classes_}
        Y_pred = model.predict(X_test)
        for i in model.classes_:
            mask = Y_test == i
            temp_y_test = Y_test[mask]
            temp_y_pred = Y_pred[mask]
            metric_dict[i]['acc'] = accuracy_score(temp_y_test, temp_y_pred)
            metric_dict[i]['prec'] = precision_score(temp_y_test, temp_y_pred)
            metric_dict[i]['rec'] = recall_score(temp_y_test, temp_y_pred)
            metric_dict[i]['f1'] = f1_score(temp_y_test, temp_y_pred)
        
        if confusion_path is not None:
            save_confusion(model, X_test, Y_test, save_path = confusion_path)
        return metric_dict
    
    def test(self, X_test, Y_test, model_path, confusion_path = None):
        self.load(model_path)
        return self.evaluate(X_test, Y_test, confusion_path = confusion_path)

    def scale_vectors(self, X_train, X_test, scaler_save = None):
        self.sc_X = StandardScaler()
        X_train = self.sc_X.fit_transform(X_train)
        X_test = self.sc_X.transform(X_test)
        if scaler_save is not None:
            with open(scaler_save, "wb") as f:
                pickle.dump(self.sc_X, f)
        return X_train, X_test

def main(no_trees = 200, max_depth = 8, dataset_path = "dataset.csv", save_model_path = "model.z", model = "adaboost"):

    X, Y, _ = get_dataset(dataset_path)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0, stratify=Y)

    classifier = Classifier(model, max_depth = max_depth, no_estimators = no_trees)

    X_train, X_test = classifier.scale_vectors(X_train, X_test, "scaler.pkl")
    classifier.train(X_train, Y_train)
    classifier.save_model(save_model_path)
    accuracy = classifier.evaluate(X_test, Y_test, confusion_path = "confusion_matrix.png")
    print("The {} classifier with {} decision trees has an accuracy of {}%".format(model, no_trees, 100*accuracy))

if __name__ == "__main__":
    main()