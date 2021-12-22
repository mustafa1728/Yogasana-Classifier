import pandas as pd 
import json
import os



main_csv_path = "../../yadav_dataset/extracted.csv"
json_root_path = "../../yadav_dataset_kps/"

main_df = pd.read_csv(main_csv_path)
print(main_df.head())
print(main_df.columns)

dataset_dict = {"image_path":[], "asana": [], "subject": [], "class": [], "frame":[], "x0": [], "y0": [], "width": [], "height": []}
for i in range(136):
	for j in range(3):
		dataset_dict["keypt" + "_" + str(i) + "_" + str(j)] = []


def fix_subjects(subject):
    if subject == "Sathak":
        return "Sarthak"
    if subject == "Shiv":
        return "Shiva"
    if subject == "veena":
        return "Veena"
    if subject == "deepa":
        return "Deepa"
    if subject == "lakshmi":
        return "Lakshmi"
    return subject
    

def fix_class_naming(asana):
    if asana == "Trik" or asana == "trikon" or asana == "Trikon" or asana == "Trikonasana":
        return "Trikonasana"
    if asana == "Bhuj" or asana == "bhuj" or asana == "bhujan" or asana == "Bhuj2" or asana == "bhujang" or asana == "Bhu": 
        return "Bhujangasana"
    if asana == "Shav" or asana == "shav" or asana == "shava" or asana == "Shavasana" or asana == "savasan": 
        return "Shavasana"
    if asana == "padam" or asana == "Padam" or asana == "padmasan": 
        return "Padamasana"
    if asana == "tadasan" or asana == "tadasna" or asana == "Tad" or asana == "Tadasan" or asana == "Tadasna" or asana == "Tadasana" or asana == "Tada": 
        return "Tadasana"
    if asana == "vriksh" or asana == "Vriksh" : 
        return "Vrikshasana"
    raise ValueError("unhandled asana {}".format(asana))

def update_dict(path, asana, subject, x0, y0, w, h, kps, frame_no):
    dataset_dict["image_path"].append(path)
    dataset_dict["asana"].append(asana)
    dataset_dict["class"].append(fix_class_naming(asana))
    dataset_dict["subject"].append(fix_subjects(subject))
    dataset_dict["x0"].append(x0)
    dataset_dict["y0"].append(y0)
    dataset_dict["width"].append(w)
    dataset_dict["height"].append(h)
    dataset_dict["frame"].append(frame_no)
    print(kps[0])
    for i in range(136):
        for j in range(3):
            dataset_dict["keypt" + "_" + str(i) + "_" + str(j)].append(kps[i*3 + j])

curated_removed = 0
total_data = 0

for i, (asana, subject, frame_no, path) in enumerate(zip(main_df["asana"], main_df["subject"], main_df["frame_no"], main_df["path"])):

    print("processing {} - {} - {}".format(asana, subject, frame_no))
    folder_name = path.split('/')[-2]
    img_name = path.split('/')[-1]

    curated_root_dir = "../../yadav_dataset"
    # if not os.path.isfile(os.path.join(curated_root_dir, folder_name, img_name)):
    #     print("removed in curation")
    #     curated_removed+=1
    #     continue

    try:
        with open(os.path.join(json_root_path, folder_name, "alphapose-results.json")) as f:
            data = json.load(f)
    except:
        print("file not found. Skipping")
        continue
    

    curr_dat = None
    for dat in data:
        if dat["image_id"] == img_name:
            curr_dat = dat

    if curr_dat is None:
        continue

    try:
        x0, y0, w, h = curr_dat["box"][0], curr_dat["box"][1], curr_dat["box"][2], curr_dat["box"][3]
    except:
        print(curr_dat["box"])
        print("some error with the bounding box here. Skipping!")
    

    
    update_dict(os.path.join(folder_name, img_name), asana, subject, x0, y0, w, h, curr_dat["keypoints"], frame_no)
    total_data += 1

pd.DataFrame(dataset_dict).to_csv("yadav_dataset_no_curation.csv", index = False)
print("{} frames removed during curation. {} frames remain.".format(curated_removed, total_data))


def process_dataset_to_get_largest_bbox(dataset_path, save_path):
    dataset = pd.read_csv(dataset_path)
    print(dataset.describe())
    dataset["size"] = dataset["width"].values * dataset["height"].values
    ids_max_bbox = dataset.groupby(['image_path'])["size"].idxmax()
    dataset = dataset.iloc[ids_max_bbox, :]
    dataset = dataset.drop(["size"], axis = 1)
    print(dataset.describe())
    dataset.to_csv(save_path, index=False)

process_dataset_to_get_largest_bbox("yadav_dataset_no_curation.csv", "yadav_dataset_no_curation.csv")