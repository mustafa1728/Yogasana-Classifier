import os
import pandas as pd
from datetime import date, datetime as dt

subj_ids = list(range(1, 52))

subj_ids_str = ["Subj"+str(item).zfill(3) for item in subj_ids]
asanas = ["Ardhachakrasana", "Garudasana", "Gorakshasana", "Katichakrasana", "Natarajasana", "Natavarasana", "Naukasana", "Padahastasana", "ParivrittaTrikonasana", "Pranamasana", "Santolanasana", "Still", "Tadasana", "Trikonasana", "TriyakTadasana", "Tuladandasana", "Utkatasana", "Virabhadrasana", "Vrikshasana"]
fps_data = {"asana" : [], "subject" : [], "fps": []}

root = "/mnt/project2/home/rahul/Yoga-Kinect/LinkedData/Camera1N/"

def get_avg_fps(timestamps_path):
    timestamps = pd.read_csv(timestamps_path, header=None)
    frame_numbers = timestamps.iloc[:, 0]
    times = timestamps.iloc[:, 1]
    avg_fps = 0
    count = 0
    for i in range(len(times) - 1):
        input_formats = ['%H:%M:%S.%f', '%H:%M:%S']
        for input_format in input_formats:
            try:
                if isinstance(times[i], str):
                    date1_obj = dt.strptime(times[i], input_format)
                else:
                    date1_obj = times[i]
                if isinstance(times[i+1], str):
                    date2_obj = dt.strptime(times[i+1], input_format)
                else:
                    date2_obj = times[i+1]
                break
            except ValueError:
                continue
        date_diff = date2_obj - date1_obj
        total_seconds = date_diff.total_seconds()
        total_frames = frame_numbers[i+1] - frame_numbers[i]

        if total_seconds>0:
            running_fps = total_frames/total_seconds
            avg_fps+=running_fps
            count+=1
    avg_fps = avg_fps / count
    return avg_fps

for subject_id in subj_ids_str:
    subj_root = os.path.join(root, subject_id)
    for asana in asanas:
        folder_name = subject_id + "_" + asana
        folder_path = os.path.join(subj_root, folder_name)
        try:
            fps = get_avg_fps(os.path.join(folder_path, "color_timestamps.csv"))
        except FileNotFoundError:
            continue

        fps_data["asana"].append(asana)
        fps_data["subject"].append(subject_id)
        fps_data["fps"].append(fps)

df = pd.DataFrame(fps_data)
df.to_csv("fps.csv", index = False)