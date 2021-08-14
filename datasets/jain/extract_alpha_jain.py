import json
import pandas as pd
from tqdm import tqdm

df = pd.read_csv('Sampled_Jain_Metadata.csv')
with open('alphapose-results-jain.json', 'r') as f:
    alpha_res = json.load(f)

last = alpha_res[0]
kp_dict = {'class':[], 'video': [], 'image_id':[]}
for i in range(136):
    kp_dict['keypt_%d_0'%i] = []
    kp_dict['keypt_%d_1'%i] = []

kp_dict['x0'] = []
kp_dict['y0'] = []
kp_dict['width'] = []
kp_dict['height'] = []

for ddict in tqdm(alpha_res[1:]) :
    if ddict['image_id'] == last['image_id']:
        if ddict['box'][2]*ddict['box'][3] > last['box'][2]*last['box'][3] :
            last = ddict
    else:
        kp_dict['class'].append(df['asana'][df['image'] == last['image_id']].iloc[0])
        kp_dict['video'].append(df['video'][df['image'] == last['image_id']].iloc[0])
        kp_dict['image_id'].append(last['image_id'])
        for i in range(136):
            kp_dict['keypt_%d_0'%i].append(last['keypoints'][3*i])
            kp_dict['keypt_%d_1'%i].append(last['keypoints'][3*i + 1])
        kp_dict['x0'].append(last['box'][0])
        kp_dict['y0'].append(last['box'][1])
        kp_dict['width'].append(last['box'][2])
        kp_dict['height'].append(last['box'][3])
        last = ddict

kp_dict['class'].append(df['asana'][df['image'] == last['image_id']].iloc[0])
kp_dict['video'].append(df['video'][df['image'] == last['image_id']].iloc[0])
kp_dict['image_id'].append(last['image_id'])
for i in range(136):
    kp_dict['keypt_%d_0'%i].append(last['keypoints'][3*i])
    kp_dict['keypt_%d_1'%i].append(last['keypoints'][3*i + 1])
kp_dict['x0'].append(last['box'][0])
kp_dict['y0'].append(last['box'][1])
kp_dict['width'].append(last['box'][2])
kp_dict['height'].append(last['box'][3])

pd.DataFrame(kp_dict).to_csv('Jain_Sampled_Dataset.csv', index=False)