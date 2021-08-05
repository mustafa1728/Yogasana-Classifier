import os
import json
import pandas as pd
import logging

original_videos_root = "/mnt/project2/home/rahul/Yoga-Kinect/VideosAll/"
root = "/home1/ee318062/"
time_stamps = pd.read_csv("timestamps.csv")
fps_df = pd.read_csv("fps.csv")
camera_data = pd.read_csv("camera.csv")


camera_mapping = {"Still": 1}
for index, row in camera_data.iterrows():
	camera_mapping[row['asana']] = row["camera"]

no_frames_per_video = 210
padding = 1
asana_id_mapping = {}


logging.basicConfig(filename="logs.txt",
					filemode='w',
					format='%(levelname)-6s | %(message)s',
					level=logging.WARNING)

final_dict = {"camera": [], "subject": [], "asana": [], "class": [], "x0": [], "y0": [], "width": [], "height": []}
for i in range(136):
	for j in range(3):
		final_dict["keypt" + "_" + str(i) + "_" + str(j)] = []

def change_convention(asana, subject_id):
	subject_number = int(subject_id[-3:])
	
	max_old = 25
	if subject_number<=max_old:
		return None, None
	subject_number -= max_old
	subject_id = subject_id[:-3] + str(subject_number).zfill(3)

	return asana, subject_id

def get_folder_name(asana):
	try:
		camera = camera_mapping[asana]
	except KeyError:
		camera = 1
		logging.error("Camera not known for asana: " + asana + "! Using camera 1")
	return "Yoga_Camera" + str(int(camera)) + "_AlphaPose"

def get_fps(asana, subject_id, prefix = "N"):
	fps = fps_df[(fps_df["asana"] == asana) & (fps_df["subject"] == subject_id)].iloc[0]["fps"]
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

def update_dict(asana, subject, camera, x0, y0, w, h):
	final_dict["asana"].append(asana)
	final_dict["subject"].append(subject)
	final_dict["camera"].append(camera)
	final_dict["x0"].append(x0)
	final_dict["y0"].append(y0)
	final_dict["width"].append(w)
	final_dict["height"].append(h)

def main():
	for i, (asana, subject_id) in enumerate(zip(time_stamps["aasana"], time_stamps["subject"])):
		if pd.isnull(subject_id) or pd.isnull(asana):
			continue
		asana, subject_id = change_convention(asana, subject_id)
		if asana is None or subject_id is None:
			continue
		asana_root = os.path.join(root, get_folder_name(asana))
		dir_name =  subject_id + "_" + asana
		try:
			with open(os.path.join(asana_root, dir_name, "alphapose-results.json")) as f:
				data = json.load(f)
		except FileNotFoundError:
			logging.error("Alphapose results not found in "+dir_name+" for camera " + str(camera_mapping[asana]) + "! Going to next video.")
			continue
		start_time = time_stamps.iloc[i]["position start (left)"]
		end_time = time_stamps.iloc[i]["position end (left)"]
		start_time_right = time_stamps.iloc[i]["position start (right)"]
		end_time_right = time_stamps.iloc[i]["position end (right)"]

		total_frames = len(data)
		
		try:
			fps = get_fps(asana, subject_id)
		except IndexError:
			logging.error("FPS for " + asana + " - " + subject_id + " not found! Going to next video.")
			continue
		logging.info(fps, i, asana, subject_id, start_time, end_time)

		none_frame_list = get_frame_no_list(fps, 0, start_time, total_frames, no_frames_per_video//20)
		if none_frame_list is not None:
			for frame_no in none_frame_list:
				final_dict["class"].append("None")
				x0, y0, w, h = data[frame_no]["box"][0], data[frame_no]["box"][1], data[frame_no]["box"][2], data[frame_no]["box"][3]
				update_dict(asana, subject_id, camera_mapping[asana], x0, y0, w, h)
				for j in range(136):
					for k in range(3):
						final_dict["keypt" + "_" + str(j) + "_" + str(k)].append(data[frame_no]["keypoints"][j*3 + k])

		frame_no_list = get_frame_no_list(fps, start_time, end_time, total_frames)
		if frame_no_list is not None:
			for frame_no in frame_no_list:
				final_dict["class"].append(get_asana_id(asana, "left"))
				x0, y0, w, h = data[frame_no]["box"][0], data[frame_no]["box"][1], data[frame_no]["box"][2], data[frame_no]["box"][3]
				update_dict(asana, subject_id, camera_mapping[asana], x0, y0, w, h)
				for j in range(136):
					for k in range(3):
						final_dict["keypt" + "_" + str(j) + "_" + str(k)].append(data[frame_no]["keypoints"][j*3 + k])

		frame_no_list = get_frame_no_list(fps, start_time_right, end_time_right, total_frames)
		if frame_no_list is not None:
			for frame_no in frame_no_list:
				final_dict["class"].append(get_asana_id(asana, "right"))
				x0, y0, w, h = data[frame_no]["box"][0], data[frame_no]["box"][1], data[frame_no]["box"][2], data[frame_no]["box"][3]
				update_dict(asana, subject_id, camera_mapping[asana], x0, y0, w, h)
				for j in range(136):
					for k in range(3):
						final_dict["keypt" + "_" + str(j) + "_" + str(k)].append(data[frame_no]["keypoints"][j*3 + k])

		

	df = pd.DataFrame(final_dict)
	df.to_csv("dataset_alphapose.csv", index = False)


main()