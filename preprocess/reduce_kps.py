import pandas as pd 
import numpy as np

def change_xy_format(x, y):
    answer = []
    for i in range(len(x)):
        answer.append(x[i])
        answer.append(y[i])
    return answer

def get_mean_min_max(x, y):
    new_x = []
    new_y = []
    # mean
    new_x.append(sum(x)/len(x))
    new_y.append(sum(y)/len(y))
    distances = [(x[i] - new_x[0]) + (y[i] - new_y[0]) for i in range(len(x))]
    # min
    new_x.append(x[distances.index(min(distances))])
    new_y.append(y[distances.index(min(distances))])
    # max
    new_x.append(x[distances.index(max(distances))])
    new_y.append(y[distances.index(max(distances))])

    return change_xy_format(new_x, new_y)

def reduce(kps_human, score_threshold = None):
    body_kp_i = [i for i in range(0, 26)]
    face_kp_i = [i for i in range(26, 94)]
    left_hand_kp_i = [i for i in range(94, 115)]
    right_hand_kp_i = [i for i in range(115, 136)]
    
    kps_x = kps_human[::3]
    kps_y = kps_human[1::3]
    kps_score = kps_human[2::3]

    # add all body kps
    reduced_kps = change_xy_format(kps_x[body_kp_i], kps_y[body_kp_i])

    def thres(x, indices, thres_val=score_threshold):
        if thres_val is None:
            thres_val = max([kps_score[i] for i in indices])/10
        return [x[i] for i in indices if kps_score[i] >= thres_val]

    # for face and both hands, get only the mean, min and max points
    reduced_kps += get_mean_min_max(thres(kps_x, face_kp_i), thres(kps_y, face_kp_i))
    reduced_kps += get_mean_min_max(thres(kps_x, left_hand_kp_i), thres(kps_y, left_hand_kp_i))
    reduced_kps += get_mean_min_max(thres(kps_x, right_hand_kp_i), thres(kps_y, right_hand_kp_i))

    return reduced_kps

def reduce_dataset(dataset_path, save_path, kps_start_id = 9):
    dataset = pd.read_csv(dataset_path)
    kps = dataset.iloc[:, kps_start_id:].values
    reduced_kps = np.asarray([reduce(kps[i, :]) for i in range(kps.shape[0])])
    reduced_dict = {}
    for i in range(kps_start_id):
        reduced_dict[dataset.columns[i]] = dataset.iloc[:, i].values
    for i in range(reduced_kps.shape[1]):
        reduced_dict["keypt_{}_{}".format(i//2, i%2)] = reduced_kps[:, i]
    pd.DataFrame(reduced_dict).to_csv(save_path, index=False)

if __name__ == "__main__":
    reduce_dataset("../yadav_dataset.csv", "yadav_kp_reduced.csv")