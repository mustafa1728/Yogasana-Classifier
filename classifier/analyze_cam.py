import pandas as pd
from utils import gen

def main():
    df = pd.read_csv("predictions_cam_wise_kinect_d8.csv")
    lab, pred = [], []
    i, k, prev = 0, 1, 0
    temp = None
    for (_, _, _, Y) in gen('../preprocess/dataset_kinect.csv'):
        if i == 4:
            temp = pd.DataFrame({"labels": lab, "predictions": pred})
            temp.to_csv('predictions_cam_wise_kinect_d8_%d.csv'%k, index=False)
            lab, pred = [], []
            k += 1
            i = 0
        lab += Y.tolist()
        pred += df['predictions'].iloc[prev: prev+len(Y.tolist())].tolist()
        prev += len(Y.tolist())
        i += 1
    
    temp = pd.DataFrame({"labels": lab, "predictions": pred})
    temp.to_csv('predictions_cam_wise_kinect_d8_%d.csv'%k, index=False)

if __name__ == '__main__':
    main()