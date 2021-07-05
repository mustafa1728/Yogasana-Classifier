import os
import json
import pandas as pd
import cv2
import numpy as np
import logging

original_videos_root = "/mnt/project2/home/rahul/Yoga-Kinect/VideosAll/"
time_stamps = pd.read_csv("timestamps.csv")
root = "/home1/ee318062/Yoga_Camera1_AlphaPose/"


no_frames_per_video = 200
padding = 1
asana_id_mapping = {}


logging.basicConfig(filename="logs.txt",
					filemode='w',
					format='%(levelname)-6s | %(message)s',
					level=logging.WARNING)

final_dict = {"class": []}
for i in range(136):
	for j in range(3):
		final_dict["keypt" + "_" + str(i) + "_" + str(j)] = []

def get_fps(asana, subject_id, prefix = "N"):
	vid_name =  "Sub"+subject_id[-3:] + prefix +"_" + asana + "_Camera1.avi"
	vid_path = os.path.join(original_videos_root, vid_name)
	cap = cv2.VideoCapture(vid_path)
	if cap is None or not cap.isOpened():
		if prefix == "N":
			logging.warning("Original video not found N, looking for O!")
			return get_fps(asana, subject_id, "O")
		else:
			logging.warning("Original video not found both N and O! Taking default 20")
			return 20.0
	fps = cap.get(cv2.CAP_PROP_FPS)
	return fps


def get_frame_no_list(fps, start_time, end_time, total_frames, fpv = no_frames_per_video):
	if pd.isnull(start_time) or pd.isnull(end_time):
		return None
	if end_time*fps >= total_frames:
		logging.warning("Data mismatch. Alphapose output json has only "+str(total_frames)+" frames. Timestamps go upto "+str(int(end_time*fps))+" frames!")
		end_time = int(total_frames/fps)
	if start_time+padding>end_time-padding:
		logging.error("Start time: "+str(start_time)+" should be less than end time: "+str(end_time))
		return None
	frame_no_list = []
	if fpv<=0:
		logging.error("No extractable frames found. FPV: "+str(fpv)+" should be positive! Going to next list.")
		return None
	stride = ((end_time - start_time - 2*padding)*fps)//fpv 
	if stride <= 0:
		logging.warning("Too less time between start and end for stride: "+str(stride)+". Reducing fpv: "+str(fpv))
		return get_frame_no_list(fps, start_time, end_time, total_frames, fpv - 10)
	for i in range(fpv):
		frame_no = (start_time + padding) * fps + i * stride
		frame_no_list.append(int( frame_no ))
	return frame_no_list
	


def get_asana_id(asana, direction):
	'''
		asana_ids will be in format asana-name_left or asana-name_right
		need to convert this into ids
	'''
	return asana + "_" + direction

for i, (asana, subject_id) in enumerate(zip(time_stamps["aasana"], time_stamps["subject"])):
	if pd.isnull(subject_id) or pd.isnull(asana):
		continue
	dir_name =  subject_id + "_" + asana
	try:
		with open(os.path.join(root, dir_name, "alphapose-results.json")) as f:
			data = json.load(f)
	except FileNotFoundError:
		logging.error("Alphapose results not found in "+dir_name+"! Going to next video.")
		continue
	start_time = time_stamps.iloc[i]["position start (left)"]
	end_time = time_stamps.iloc[i]["position end (left)"]
	start_time_right = time_stamps.iloc[i]["position start (right)"]
	end_time_right = time_stamps.iloc[i]["position end (right)"]

	total_frames = len(data)
	
	fps = get_fps(asana, subject_id)
	logging.info(fps, i, asana, subject_id, start_time, end_time)

	none_frame_list = get_frame_no_list(fps, 0, start_time, total_frames, no_frames_per_video//20)
	if none_frame_list is not None:
		for frame_no in none_frame_list:
			final_dict["class"].append("None")
			for j in range(136):
				for k in range(3):
					final_dict["keypt" + "_" + str(j) + "_" + str(k)].append(data[frame_no]["keypoints"][j*3 + k])

	frame_no_list = get_frame_no_list(fps, start_time, end_time, total_frames)
	if frame_no_list is not None:
		for frame_no in frame_no_list:
			final_dict["class"].append(get_asana_id(asana, "left"))
			for j in range(136):
				for k in range(3):
					final_dict["keypt" + "_" + str(j) + "_" + str(k)].append(data[frame_no]["keypoints"][j*3 + k])

	frame_no_list = get_frame_no_list(fps, start_time_right, end_time_right, total_frames)
	if frame_no_list is not None:
		for frame_no in frame_no_list:
			final_dict["class"].append(get_asana_id(asana, "right"))
			for j in range(136):
				for k in range(3):
					final_dict["keypt" + "_" + str(j) + "_" + str(k)].append(data[frame_no]["keypoints"][j*3 + k])


df = pd.DataFrame(final_dict)
df.to_csv("dataset.csv", index = False)
