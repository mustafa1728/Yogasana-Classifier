from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
#from model import Classifier
from lightgbm import LGBMClassifier

df = pd.read_csv('../preprocess/Jain_Sampled_Dataset.csv')

Y = df['class']
X1 = np.array(df.iloc[:, 3:-4])
bbox = np.array(df.iloc[:, -4:])

X1[:, [i%2 == 0 for i in range(272)]] = (X1[:, [i%2 == 0 for i in range(272)]] - bbox[:, 0].reshape(-1, 1))/bbox[:, 2].reshape(-1, 1)
X1[:, [i%2 != 0 for i in range(272)]] = (X1[:, [i%2 != 0 for i in range(272)]] - bbox[:, 1].reshape(-1, 1))/bbox[:, 3].reshape(-1, 1)
X2 = (bbox[:, 2]/bbox[:, 3]).reshape(-1, 1)

X = np.concatenate([X1, X2], axis=1)

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2)

predictions, labels = [], []
i = 1
for train_index, test_index in kf.split(X, Y):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    #classifier = Classifier('random_forest', max_depth = None, no_estimators = 500)
    rf = RandomForestClassifier(n_estimators=500, criterion='gini', random_state=1)
    hg = HistGradientBoostingClassifier(random_state=7)
    lgbm = LGBMClassifier(boosting_type='goss', n_estimators = 500, random_state = 71, learning_rate=0.1)
    vc = VotingClassifier([('rf', rf), ('hg', hg), ('lg',lgbm)], voting='soft', weights=[1, 2, 3])
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    #X_train, X_test = classifier.scale_vectors(X_train, X_test, "scaler.pkl")
    #classifier.train(X_train, Y_train)
    vc.fit(X_train, Y_train)

    Y_pred = vc.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred)
    print('Accuracy in fold %d : %.2f'%(i, acc*100.0)+' %')
    i += 1
    predictions.append(Y_pred)
    labels.append(Y_test)

df_predictions = pd.DataFrame({"labels": np.concatenate(labels), "predictions": np.concatenate(predictions)})
df_predictions.to_csv("predictions_frame_wise_jain.csv", index = False)