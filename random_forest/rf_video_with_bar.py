from ..utils import create_video_with_proba
from inference import load_model
import json

model_path=''
video_path=''
key_pt_path=''

id_to_class_map = None
with open('ids_to_class.json', 'r') as f:
    id_to_class_map = json.load(f)

save_folder = './Bar_Plots'
classifier = load_model(model_path)

create_video_with_proba(video_path, classifier, key_pt_path, id_to_class_map, save_folder)