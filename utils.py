from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import json
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import plot_confusion_matrix
import pandas as pd

def load_model(model_weights_path):
    classifier = joblib.load(model_weights_path)
    return classifier

def get_dataset(dataset_path):
    dataset = pd.read_csv(dataset_path)
    dataset = pre_process_labels(dataset)
    indices_to_keep = ~dataset.isin([np.nan, np.inf, -np.inf]).any(1)
    dataset = dataset[indices_to_keep]
    X = dataset.iloc[:, 1:].values
    Y = dataset.iloc[:, 0].values
    return X, Y

def save_confusion(classifier, X_Test, Y_Test, display_labels=None, save_path = "confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(20, 16))
    plot_confusion_matrix(
        classifier, 
        X_Test, Y_Test, ax=ax,
        display_labels=display_labels, 
        cmap=plt.cm.Blues,
        normalize="pred",
        xticks_rotation = "vertical"
    )
    plt.savefig(save_path, dpi = 300)

def pre_process_labels(dataset):
    class_to_id_mapping = {} 
    id_to_class_mapping = {}

    classes = list(dataset["class"].unique())
    for i in range(len(classes)):
        class_to_id_mapping[classes[i]] = i
        id_to_class_mapping[i] = classes[i]
    dataset["class"] = dataset["class"].apply(lambda x:class_to_id_mapping[x])
    return dataset, class_to_id_mapping, id_to_class_mapping

def create_train_test(X, Y, test_size):
    # Splitting the dataset into the Training set and Test set
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = test_size, random_state = 0)
    
    # Feature Scaling
    sc_X = StandardScaler()
    X_Train = sc_X.fit_transform(X_Train)
    X_Test = sc_X.transform(X_Test)

    return X_Train, X_Test, Y_Train, Y_Test

#def get_parent_args():
#    parent_args = args.ArgumentParser(add_help=False)
#    parent_args.add_argument('--save_model_path', '-s', dest='save_model_path', nargs=1, help='Specify path to save trained model')
#    parent_args.add_argument('--model_path', '-m', dest='model_path', nargs=1, help='Specify the Trained Model Path')
#    parent_args.add_argument('--data', '-d', dest='data', nargs=1, help='Specify path to the dataset .csv file')
#    parent_args.add_argument('--results', '-r', nargs=1, dest='res', help='File path to store the results')
#    return parent_args

def create_video_with_proba(video_path, classifier, path_to_kp, id_to_class_mapping, save_folder):
    key_pts = None
    with open(path_to_kp, 'r') as f:
        key_pts = json.load(f)
    classes = classifier.classes_
    class_names = [id_to_class_mapping[c] for c in classes]
    y_pos = np.arange(len(classes))

    for k in range(len(key_pts)):
        plt.barh(y_pos, classifier.predict_proba(np.array(key_pts[k]['keypoints']).reshape(1, -1)).reshape(-1), align='center')
        plt.set_yticks(y_pos)
        plt.set_yticklabels(class_names)
        plt.invert_yaxis()
        plt.set_xlabel('Probability')
        plt.set_title('Yogasana Classifier')
        plt.savefig(os.path.join(save_folder, 'prob_{}.png'.format(k)), dpi=400)
    
    fps = 0
    vc = cv.VideoCapture(video_path)
    (major_ver, _, _) = (cv.__version__).split('.')
    if int(major_ver) < 3:
        fps = vc.get(cv.cv.CV_CAP_PROP_FPS)
    else:
        fps = vc.get(cv.CAP_PROP_FPS)
    
    fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
    succ, fr = vc.read()
    height, width = fr.shape
    p_height, p_width = int(0.17*height), int(0.83*width)
    height -= p_height

    name = os.path.basename(video_path)[:-4]
    video = cv.VideoWriter('{}_Output.mp4'.format(name), fourcc, float(fps), (height+p_height, p_width))
    i = 0
    while succ:
        fr = cv.resize(fr, dsize=(height, p_width), interpolation=cv.INTER_AREA)
        bar = cv.imread(os.path.join(save_folder, 'prob_{}.png'.format(i)))
        bar = cv.resize(bar, dsize=(p_height, p_width), interpolation=cv.INTER_AREA)
        res = cv.vconcat([bar, fr])
        video.write(res)
        i = i + 1
        succ, fr = vc.read()
    
    video.release()  