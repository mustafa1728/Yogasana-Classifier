import os
import cv2
#import json
import pandas as pd
import logging

original_videos_root = "/mnt/project/Yoga-Kinect/"
%root = "/home1/ee318062/"
time_stamps = pd.read_csv("timestamps.csv")
fps_df = pd.read_csv("fps.csv")
camera_data = pd.read_csv("camera.csv")
vid_fname = "color.avi"
save_dir = "/mnt/local/YogaPoseEstimation/all_videos_extracted_frames/"

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

final_dict = {"video_path": [], "frame_no": [], "class": []}
#for i in range(136):
#	for j in range(3):
#		final_dict["keypt" + "_" + str(i) + "_" + str(j)] = []

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
		logging.warning("Data mismatch. Video has only "+str(total_frames)+" frames. Timestamps go upto "+str(int(end_time*fps))+" frames!")
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


def update_dict(frame_no, video_path):
    final_dict["frame_no"].append(frame_no)
    final_dict["video_path"].append(video_path)


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
            pth = os.path.join(asana_root, dir_name, vid_fname)
			cap = cv2.VideoCapture(pth)
		except FileNotFoundError:
			logging.error("Video File not found in "+dir_name+" for camera " + str(camera_mapping[asana]) + "! Going to next video.")
			continue
		start_time = time_stamps.iloc[i]["position start (left)"]
		end_time = time_stamps.iloc[i]["position end (left)"]
		start_time_right = time_stamps.iloc[i]["position start (right)"]
		end_time_right = time_stamps.iloc[i]["position end (right)"]

		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		
		try:
			fps = get_fps(asana, subject_id)
		except IndexError:
			logging.error("FPS for " + asana + " - " + subject_id + " not found! Going to next video.")
			continue
		logging.info(fps, i, asana, subject_id, start_time, end_time)
        i = 1
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video = cv2.VideoWriter(os.path.join(save_dir, pth[:-4]+"_sampled"+pth[-4:], fourcc, 1, (width, height)))
        none_frame_list = get_frame_no_list(fps, 0, start_time, total_frames, no_frames_per_video//20)
        frame_no_list_left = get_frame_no_list(fps, start_time, end_time, total_frames)
        frame_no_list_right = get_frame_no_list(fps, start_time_right, end_time_right, total_frames)

        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                logging.error(pth + 'abruptly ended at frame' + str(i) + '! Going to next video.')
                break
                
            if none_frame_list is not None:
                if i in none_frame_list:
                    final_dict["class"].append("None")
                    update_dict(i, pth[:-4]+"_sampled"+pth[-4:])
                    video.wrtie(frame)

		    if frame_no_list_left is not None:
                if i in frame_no_list_left:
				    final_dict["class"].append(get_asana_id(asana, "left"))
                    update_dict(i, pth[:-4]+"_sampled"+pth[-4:])
                    video.wrtie(frame)
				
		    if frame_no_list_right is not None:
                if i in frame_no_list_right:
				    final_dict["class"].append(get_asana_id(asana, "right"))
                    update_dict(i, pth[:-4]+"_sampled"+pth[-4:])
                    video.wrtie(frame)
            
        cap.release()
        video.release()

		

	df = pd.DataFrame(final_dict)
	df.to_csv("frame_dataset.csv", index = False)


main()