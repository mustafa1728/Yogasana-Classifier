#from ..utils import create_video_with_proba
#from inference import load_model
import json
import os
import pickle
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import joblib

model_path='model.z'
scaler_path=''
video_path='/home1/ee318062/Yoga_Camera1_AlphaPose/Subj001_Ardhachakrasana/AlphaPose-color.avi'
key_pt_path='/home1/ee318062/Yoga_Camera1_AlphaPose/Subj001_Ardhachakrasana/alphapose-results.json'
frames_folder='/home1/ee318062/Yoga_Camera1_AlphaPose/Subj001_Ardhachakrasana/vis'

id_to_class_map = None
with open('ids_to_class.json', 'r') as f:
    id_to_class_map = json.load(f)

ls = os.path.dirname(video_path).split('/')[3:5]
cam = ls[0].split('_')[1]
name = ls[1] + '_' + cam
save_folder = './'+name+'_Bar_Plots'
save_folder_combined = './'+name+'_Combined'
classifier = joblib.load(model_path)
scaler = joblib.load(scaler_path)

def resize_pad(img, width, height, interpolation=cv.INTER_AREA):
    curr_img = cv.resize(img, ((img.shape[1]*height)//img.shape[0], height), interpolation=interpolation)    
    final_img = np.full((height,width,3), (255, 255, 255), dtype=np.uint8)
    center1 = (height - curr_img.shape[0])//2
    center2 = (width - curr_img.shape[1])//2
    final_img[center1:center1 + curr_img.shape[0], center2:center2 + curr_img.shape[1], :] = curr_img
    return final_img

def create_video_with_proba(video_path, frames_folder, classifier, scaler, path_to_kp, id_to_class_mapping, save_folder, save_folder_combined):
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    if not os.path.isdir(save_folder_combined):
        os.mkdir(save_folder_combined)
    key_pts = None
    with open(path_to_kp, 'r') as f:
        key_pts = json.load(f)
    classes = classifier.classes_
    class_names = [id_to_class_mapping[str(c)] for c in classes]
    y_pos = np.arange(len(classes))

    for k in range(len(key_pts)):
        plt.rcdefaults()
        fig, ax = plt.subplots()
        ax.barh(y_pos, classifier.predict_proba(sclaer.transform(np.array(key_pts[k]['keypoints']).reshape(1, -1))).reshape(-1), align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(class_names)
        ax.invert_yaxis()
        ax.set_xlabel('Probability')
        ax.set_title('Yogasana Classifier')
        fig.savefig(os.path.join(save_folder, 'prob_{}.png'.format(k)), dpi=400, bbox_inches='tight')
        plt.close(fig)
        #print(k, 'bar plots obtained')
        if k >= 10:
           break
    i = 0
    while os.path.isfile(os.path.join(frames_folder, '{}.jpg'.format(i))):
        
        fr = cv.imread(os.path.join(frames_folder, '{}.jpg'.format(i)))
        bar = cv.imread(os.path.join(save_folder, 'prob_{}.png'.format(i)))
        bar = resize_pad(bar, width = fr.shape[1], height = fr.shape[0]//2, interpolation=cv.INTER_AREA)
        #print(bar.shape, fr.shape)
        res = cv.vconcat([bar, fr])
        cv.imwrite(os.path.join(save_folder_combined, 'combined_{}.jpg'.format(i)), res)
        i = i + 1
        if i >= 10:
           break

create_video_with_proba(video_path, frames_folder, classifier, scaler, key_pt_path, id_to_class_map, save_folder, save_folder_combined)
