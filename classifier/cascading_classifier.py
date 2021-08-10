import pandas as pd
from model import Classifier
import numpy as np


class CascadingClassifier():
    def __init__(self, method, max_depth=None, no_estimators=500, lr=0.05, random_state=0):
        self.method = method
        self.classifier1 = Classifier(method, max_depth, no_estimators, lr, random_state)
        self.classifier2 = Classifier(method, max_depth, no_estimators, lr, random_state)
        self.classifier3 = Classifier(method, max_depth, no_estimators, lr, random_state)
        self.pred_1 = self.pred_2 = self.pred_3 = None
        
    def train_level_1(self, kp_ar_feat, Y):
        print("Training classifier 1")
        self.classifier1.train(kp_ar_feat, Y)
        self.pred_1 = self.classifier1.model.predict(kp_ar_feat)

    def train_level_2(self, kp_ar_feat, Y, pred_1):
        print("Training classifier 2")
        feat = np.append(kp_ar_feat, pred_1.reshape(-1, 1), axis = 1)
        self.classifier2.train(feat, Y)
        self.pred_2 = self.classifier2.model.predict(feat)

    def train_level_3(self, kp_ar_feat, Y, pred_1, pred_2):
        print("Training classifier 3")
        feat = np.append(kp_ar_feat, pred_1.reshape(-1, 1), axis = 1)
        feat = np.append(feat, pred_2.reshape(-1, 1), axis = 1)
        self.classifier3.train(feat, Y)
        self.pred3 = self.classifier3.model.predict(feat)

    def train(self, kp_ar_feat, Y1, Y2, Y3):
        self.train_level_1(kp_ar_feat, Y1)
        self.train_level_2(kp_ar_feat, Y2, self.pred_1)
        self.train_level_3(kp_ar_feat, Y3, self.pred_1, self.pred_2)

    def evaluate(self, kp_ar_feat, Y1, Y2, Y3, save_predictions_path = None):
        feat = kp_ar_feat
        self.accuracy1 = self.classifier1.evaluate(feat, Y1)
        print("Level 1 classification achieves {:.2f}% accuracy".format(self.accuracy1*100))
        self.pred_1 = self.classifier2.model.predict(feat)
        feat = np.append(feat, self.pred_1.reshape(-1, 1), axis = 1)
        self.accuracy2 = self.classifier2.evaluate(feat, Y2)
        print("Level 2 classification achieves {:.2f}% accuracy".format(self.accuracy2*100))
        self.pred_2 = self.classifier3.model.predict(feat)
        feat = np.append(feat, self.pred_2.reshape(-1, 1), axis = 1)
        self.accuracy3 = self.classifier3.evaluate(feat, Y3)
        print("Level 3 classification achieves {:.2f}% accuracy".format(self.accuracy3*100))
        self.pred_3 = self.classifier3.model.predict(feat)
        if save_predictions_path is not None:
            summary = {"labels1": Y1, "labels2": Y2, "labels3": Y3, "pred1": self.pred_1, "pred2": self.pred_2, "pred3": self.pred_3}
            pd.DataFrame(summary).to_csv(save_predictions_path, index=False)

def get_Y(data, train_df_path="/Users/mustafa/Desktop/yoga/Yoga-82/yoga_train.txt", test_df_path = "/Users/mustafa/Desktop/yoga/Yoga-82/yoga_test.txt"):
    train_df = pd.read_csv(train_df_path, sep=',', lineterminator='\n', header = None)
    test_df = pd.read_csv(test_df_path, sep=',', lineterminator='\n', header = None)
    paths = data.iloc[:, 0].values
    Y1 = []
    Y2 = []
    Y3 = []
    train_paths = train_df.iloc[:, 0].values
    test_paths = test_df.iloc[:, 0].values
    for pth in paths:
        found = False
        for i in range(train_paths.shape[0]):
            if train_paths[i] == pth:
                Y1.append(train_df.iloc[i, 1])
                Y2.append(train_df.iloc[i, 2])
                Y3.append(train_df.iloc[i, 3])
                found = True
                break
        if not found:
            for i in range(test_paths.shape[0]):
                if test_paths[i] == pth:
                    Y1.append(test_df.iloc[i, 1])
                    Y2.append(test_df.iloc[i, 2])
                    Y3.append(test_df.iloc[i, 3])
                    found = True
                    break
    return np.asarray(Y1), np.asarray(Y2), np.asarray(Y3)
    
def normalise(data, X):
    corner_x = data.iloc[:, 2].values
    corner_y = data.iloc[:, 3].values
    widths = data.iloc[:, 4].values
    heights = data.iloc[:, 5].values
    for i in range(136):
        X[:, 2*i] = (X[:, 2*i] - corner_x) * (1 / widths)
        X[:, 2*i+1] = (X[:, 2*i+1] - corner_y) * (1 / heights)
    return X


def main():
    dataset_path = "../yoga_82_max_bbox_kponly.csv"
    dataset = pd.read_csv(dataset_path)
    classifier = CascadingClassifier("random_forest")

    Y1, Y2, Y3 = get_Y(dataset)
    X = dataset.iloc[:, 6:].values
    bbox = dataset.iloc[:, 4:6].values
    X = normalise(dataset, X)
    as_ratios = bbox[:, 1] / bbox[:, 0]
    kp_ar_feat = np.append(X, as_ratios.reshape(-1, 1), axis = 1)
    classifier.train(kp_ar_feat, Y1, Y2, Y3)

main()