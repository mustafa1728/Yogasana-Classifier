import os
import pandas as pd
import logging
import numpy as np

original_videos_root = "/mnt/project2/home/rahul/Yoga-Kinect/VideosAll/"
root = "/home1/ee318062/"
time_stamps = pd.read_csv("timestamps.csv")

meta_csv = pd.read_csv("Kinect_Metadata_Full.csv")

# print(combined_csv.columns)

no_frames_per_video = 210
padding = 1
asana_id_mapping = {}


logging.basicConfig(filename="logs_kinect.txt",
					filemode='w',
					format='%(levelname)-6s | %(message)s',
					level=logging.INFO)

final_dict = {"subject": [], "camera": [], "asana": [], "class": []}
for i in range(25):
	for j in range(2):
		final_dict["keypt" + "_" + str(i) + "_" + str(j)] = []

def get_timestamp_convention(subject_id, suffix):
    if suffix == "O":
        return subject_id
    elif suffix == "N":
        subject_number = int(subject_id[-3:])
        subject_number += 25
        return subject_id[:-3] + str(subject_number).zfill(3)
	#return subject_id

def get_fps(i):
	# fps = combined_csv[(combined_csv["asana"] == asana) & (combined_csv["subject"] == subject_id)].iloc[0]["fps"]
	no_frames = meta_csv.iloc[i]["Num_Frames"]
	total_time = meta_csv.iloc[i]["Duration"]
	total_time = float(total_time)
	return no_frames/total_time, no_frames


def get_frame_no_list(fps, start_time, end_time, total_frames, fpv = no_frames_per_video):
	if pd.isnull(start_time) or pd.isnull(end_time):
		return None
	if end_time*fps >= total_frames:
		logging.warning("Data mismatch. Kinect joints.csv has only "+str(total_frames)+" frames. Timestamps go upto "+str(int(end_time*fps))+" frames!")
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
	

def get_keypoint(joints, frame_no, i, j):
	current_frame_df = joints[(joints[4] == frame_no)]
	try:
		keypoint = current_frame_df.iloc[i][5+j]
	except IndexError: 
		keypoint = np.nan
	return keypoint


def get_asana_id(asana, direction):
	'''
		asana_ids will be in format asana-name_left or asana-name_right
		need to convert this into ids
	'''
	return asana + "_" + direction

def update_dict(asana, subject, cam):
	final_dict["asana"].append(asana)
	final_dict["subject"].append(subject)
	final_dict["camera"].append(cam)

def main():
	#for i, (file_path, subject_id, asana, suffix) in enumerate(zip(meta_csv["Path"], meta_csv["Subject"], meta_csv["Asana"], meta_csv["O/N"])):
	#	if pd.isnull(file_path) or pd.isnull(subject_id) or pd.isnull(asana) or pd.isnull(suffix):
	#		logging.error("Missing data at index " + str(i) + "! Going to next video.")
	#		continue
		
		# print(os.readlink(dir_path), os.path.isfile(os.path.join(os.readlink(dir_path), "joints.csv")))
		#try:
		#	joints = pd.read_csv(os.path.join(dir_path, "joints.csv"), header=None)
		#except FileNotFoundError:
		#	logging.error("Joints data not found in " + dir_path + "! Going to next video.")
		#	continue
	with open("index.txt", "r") as f:
		i = int(f.read())
	
	for row in meta_csv.itertuples(name=None):
		if(row[0] < i):
			continue
		subject_id = get_timestamp_convention(row[1], row[2])
		asana = row[3]
		cam = row[4]
		joints = pd.read_csv(row[5], header=None)
		try:
			current_dict = time_stamps[(time_stamps["aasana"] == asana) & (time_stamps["subject"] == subject_id)].iloc[0]
		except IndexError:
			logging.error("No timestamps found for "+asana+"-"+subject_id)
			continue
		start_time = current_dict["position start (left)"]
		end_time = current_dict["position end (left)"]
		start_time_right = current_dict["position start (right)"]
		end_time_right = current_dict["position end (right)"]
		
		try:
			fps, total_frames = get_fps(row[0])
		except IndexError:
			logging.error("FPS for " + asana + " - " + subject_id + " not found! Going to next video.")
			continue
		string_status = str(row[0]) + "| fps: "+str(fps)+" asana: "+asana+" subject: "+subject_id+" suffix: "+row[2]
		logging.info(string_status)
		print(string_status)

		frame_no_list = get_frame_no_list(fps, start_time, end_time, total_frames)
		if frame_no_list is not None:
			for frame_no in frame_no_list:
				final_dict["class"].append(get_asana_id(asana, "left"))
				update_dict(asana, subject_id, cam)
				for j in range(25):
					for k in range(2):
						final_dict["keypt" + "_" + str(j) + "_" + str(k)].append(get_keypoint(joints, frame_no, j, k))

		frame_no_list = get_frame_no_list(fps, start_time_right, end_time_right, total_frames)
		if frame_no_list is not None:
			for frame_no in frame_no_list:
				final_dict["class"].append(get_asana_id(asana, "right"))
				update_dict(asana, subject_id, cam)
				for j in range(25):
					for k in range(2):
						final_dict["keypt" + "_" + str(j) + "_" + str(k)].append(get_keypoint(joints, frame_no, j, k))

		none_frame_list = get_frame_no_list(fps, 0, start_time, total_frames, no_frames_per_video//20)
		if none_frame_list is not None:
			for frame_no in none_frame_list:
				final_dict["class"].append("None")
				update_dict(asana, subject_id, cam)
				for j in range(25):
					for k in range(2):
						final_dict["keypt" + "_" + str(j) + "_" + str(k)].append(get_keypoint(joints, frame_no, j, k))

		if (row[0]+1-i)%20 == 0:
			df = pd.DataFrame(final_dict)
			df.to_csv("dataset_kinect.csv", index = False)
			with open("index.txt", "w+") as f:
				f.write(str(row[0]))
			print("Checkpoint saved. %d files processed"%(row[0]+1))
		



main()