import os
import json
import pandas as pd
import cv2
import numpy as np

folder_name = ""
original_videos_root = "~/"
time_stamps = pd.read_csv("timestamps.csv")
root = "."


no_frames_per_video = 200
padding = 1
asana_id_mapping = {}

final_dict = {"class": []}
for i in range(136):
	for j in range(3):
		final_dict["keypt" + "_" + str(i) + "_" + str(j)] = []

def get_fps(asana, subject_id):
	vid_name =  subject_id + "_" + asana + "_Camera1.avi"
	cap = cv2.VideoCapture(os.path.join(original_videos_root, vid_name))
	fps = cap.get(cv2.CAP_PROP_FPS)
	return fps


def get_frame_no_list(fps, start_time, end_time, fpv = no_frames_per_video):
	if np.isnan(start_time) or np.isnan(end_time):
		return None
	frame_no_list = []
	avg_time = int((start_time + start_time)//2)
	stride = ((end_time - start_time - 2*padding)*fps)//fpv
	if stride == 0:
		return get_frame_no_list(fps, start_time, end_time, fpv - 10)
	for i in range(fpv)
		frame_no = (start_time + padding) * fps + i * stride
		frame_no_list.append( frame_no  )
	return frame_no_list
	


def get_asana_id(asana, direction):
	'''
		asana_ids will be in format asana-name_left or asana-name_right
		need to convert this into ids
	'''
	return asana_id_mapping[asana + "_" + direction]

for i, asana, subject_id in enumerate(zip(time_stamps["aasana"], time_stamps["subject"])):
	dir_name =  subject_id + "_" + asana
	with open(os.path.join(root, dir_name, "alphapose-results.json")) as f:
		data = json.load(f)
	start_time = time_stamps["position start (left)"].iloc(i)
	end_time = time_stamps["position end (left)"].iloc(i)
	start_time_right = time_stamps["position start (right)"].iloc(i)
	end_time_right = time_stamps["position end (right)"].iloc(i)
	fps = get_fps(asana, subject_id)

	none_frame_list = get_frame_no_list(fps, 0, start_time, no_frames_per_video//20)
	if none_frame_list is not None:
		for frame_no in frame_no_list:
			final_dict["class"].append("None")
			for j in range(136):
				for k in range(3):
					final_dict["keypt" + "_" + str(j) + "_" + str(k)].append(data[frame_no]["keypoints"][j*3 + k])

	frame_no_list = get_frame_no_list(fps, start_time, end_time)
	if frame_no_list is not None:
		for frame_no in frame_no_list:
			final_dict["class"].append(get_asana_id(asana, "left"))
			for j in range(136):
				for k in range(3):
					final_dict["keypt" + "_" + str(j) + "_" + str(k)].append(data[frame_no]["keypoints"][j*3 + k])

	frame_no_list = get_frame_no_list(fps, start_time_right, end_time_right)
	if frame_no_list is not None:
		for frame_no in frame_no_list:
			final_dict["class"].append(get_asana_id(asana, "right"))
			for j in range(136):
				for k in range(3):
					final_dict["keypt" + "_" + str(j) + "_" + str(k)].append(data[frame_no]["keypoints"][j*3 + k])


df = pd.DataFrame(final_dict)
df.to_csv("dataset.csv", index = False)
