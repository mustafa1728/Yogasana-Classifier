import os
import pandas as pd
import cv2


def update_dict(extracted_dict, frame_no, path, folder_name):
	subject, asana = folder_name.split("_")
	extracted_dict["path"].append(path)
	extracted_dict["asana"].append(asana)
	extracted_dict["subject"].append(subject)
	extracted_dict["frame_no"].append(frame_no)
	return extracted_dict


def extract_frames(root_dir, save_dir, no_frames_to_extract = 200, buffer_frames = 100):
	extracted_dict = {"asana": [], "subject": [], "frame_no": [], "path": []}

	files_list = [x for x in os.walk(root_dir)]
	files_list = files_list[-1][-1]
	print(len(files_list))
	print(files_list)
	frame_no = 1
	for filename in files_list:
		folder_name = filename[:-4]
		save_folder = os.path.join(save_dir, folder_name)
		os.makedirs(save_folder, exist_ok=True)
		video_path = os.path.join(root_dir, filename)
		cap= cv2.VideoCapture(video_path)
		print("extracting " + folder_name)
		frame_no=buffer_frames
		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		# stride = (total_frames - 2*buffer_frames) // no_frames_to_extract
		stride = 1
		counter = 0
		extracted_frames = 0
		while(cap.isOpened()):
			ret, frame = cap.read()
			if ret == False:
				break
			counter+=1
			if counter == frame_no and frame_no < total_frames - buffer_frames:
				path_to_save = "{}_frame_{}.jpg".format(folder_name, frame_no)
				path_to_save = os.path.join(save_folder, path_to_save)
				cv2.imwrite(path_to_save, frame)

				extracted_dict = update_dict(extracted_dict, frame_no, path_to_save, folder_name)

				frame_no += stride
				extracted_frames += 1
				# if extracted_frames>=no_frames_to_extract:
				# 	break

		cap.release()
		extracted_df = pd.DataFrame(extracted_dict)
		extracted_df.to_csv(os.path.join(save_dir, "extracted.csv"), index=False)
	
			

extract_frames("Yoga_Vid_Collected", "extracted_frames")