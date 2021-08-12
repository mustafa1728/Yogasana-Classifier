import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
import joblib
import pickle

from utils import save_confusion

class Classifier():
    '''
        Generic classifier providing unified API for different classifiers
        Currently supported methods:
            adaboost
            random_forest
            bagging
            grad_boost
            hist_grad_boost
            lgbm
            ensemble (a voting classifier of random_forest, hist_grad_boost and lgbm)

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
            self.model = RandomForestClassifier(n_estimators = self.no_estimators, criterion = 'gini', random_state = self.random_state, max_depth=self.max_depth)
        elif self.method == "bagging":
            self.model = BaggingClassifier(DecisionTreeClassifier(criterion='gini', max_depth=self.max_depth), n_estimators=self.no_estimators, random_state=self.random_state)
        elif self.method == "grad_boost":
            self.model = GradientBoostingClassifier(n_estimators = self.no_estimators, criterion = 'friedman_mse', random_state = self.random_state, max_depth=self.max_depth, learning_rate=self.lr)
        elif self.method == "hist_grad_boost":
            self.model = HistGradientBoostingClassifier(max_iter = self.no_estimators, random_state = self.random_state, max_depth=self.max_depth, learning_rate=self.lr)
        elif self.method == "lgbm":
             self.model = LGBMClassifier(boosting_type='goss', max_depth=None, n_estimators = self.no_estimators, random_state = self.random_state, learning_rate=self.lr)
        elif self.method == "ensemble":
            clf1 = RandomForestClassifier(n_estimators = self.no_estimators, criterion = 'gini', random_state = self.random_state, max_depth=self.max_depth)
            clf2 = HistGradientBoostingClassifier(max_iter = self.no_estimators, random_state = self.random_state, max_depth=self.max_depth, learning_rate=0.1)
            clf3 = LGBMClassifier(boosting_type='goss', max_depth=None, n_estimators = self.no_estimators, random_state = self.random_state, learning_rate=0.1)
            estimators=[('rf', clf1), ('lgbm', clf2), ('hgrad', clf3)]
            self.model = VotingClassifier(estimators, voting='soft', weights=[1.5,2,3])
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
        if confusion_path is not None:
            save_confusion(model, X_test, Y_test, save_path = confusion_path)
        if self.method == "LGBM":
            return accuracy_score(Y_test, self.model.predict(X_test))
        else:
            return model.score(X_test, Y_test)
    
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
