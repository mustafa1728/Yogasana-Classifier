#from ..utils import create_video_with_proba
#from inference import load_model
import json
import os
import pickle
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

model_path='model.pkl'
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
classifier = None
with open(model_path, 'rb') as f:
    classifier = pickle.load(f)

def create_video_with_proba(video_path, frames_folder, classifier, path_to_kp, id_to_class_mapping, save_folder, save_folder_combined):
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
        ax.barh(y_pos, classifier.predict_proba(np.array(key_pts[k]['keypoints']).reshape(1, -1)).reshape(-1), align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(class_names)
        ax.invert_yaxis()
        ax.set_xlabel('Probability')
        ax.set_title('Yogasana Classifier')
        fig.savefig(os.path.join(save_folder, 'prob_{}.png'.format(k)), dpi=400, bbox_inches='tight')
        plt.close(fig)
        #print(k, 'bar plots obtained')
        #if k == 500:
        #    break

    
    fps = 20
    #vc = cv.VideoCapture(video_path)
    #(major_ver, _, _) = (cv.__version__).split('.')
    #if int(major_ver) < 3:
    #    fps = vc.get(cv.cv.CV_CAP_PROP_FPS)
    #else:
    #    fps = vc.get(cv.CAP_PROP_FPS)
    
    fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
    #succ, fr = vc.read()
    height, width = 1080, 1920
    p_height, p_width = int(0.17*height), int(0.83*width)
    height -= p_height

    ls = os.path.dirname(video_path).split('/')[3:5]
    cam = ls[0].split('_')[1]
    name = ls[1] + '_' + cam

    video = cv.VideoWriter('{}_Output.mp4'.format(name), fourcc, float(fps), (height+p_height, p_width))
    i = 0
    while os.path.isfile(os.path.join(frames_folder, '{}.jpg'.format(i))):
        fr = cv.imread(os.path.join(frames_folder, '{}.jpg'.format(i)))
        fr = cv.resize(fr, dsize=(p_width, height), interpolation=cv.INTER_AREA)
        bar = cv.imread(os.path.join(save_folder, 'prob_{}.png'.format(i)))
        bar = cv.resize(bar, dsize=(p_width, p_height), interpolation=cv.INTER_AREA)
        #print(bar.shape, fr.shape)
        res = cv.vconcat([bar, fr])
        cv.imwrite(os.path.join(save_folder_combined, 'combined_{}.jpg'.format(i)), res)
        video.write(res)
        i = i + 1
        #if i == 500:
        #    break
    video.release()

create_video_with_proba(video_path, frames_folder, classifier, key_pt_path, id_to_class_map, save_folder, save_folder_combined)
