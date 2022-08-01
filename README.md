# Yogasana-Classifier

Official codebase for the paper "A View Independent Classification Framework for Yoga Postures".

### Installation

Packages required are listed in [requirements](./requirements.txt). There is a dependency on LighGBM ([paper](https://ui.adsabs.harvard.edu/abs/2021arXiv210900724Y/abstract)) ([docs](https://lightgbm.readthedocs.io/en/latest/)), and rest of the packages are standard libraries.

### Datasets

The processed key point inferred datasets can be found [here](https://drive.google.com/drive/folders/13xcSisb_UNwVRhV_joR0rXG3kmK_DDVi?usp=sharing).

### Code Overview

Preprocessing codes to generate key points dataset from raw video data and alhpapose key points can be found [here](./preprocess). It also includes code for bounding box normalisation, key point selection and fold generation. 

The main classifier class can be found [here](./classifier/model.py). It contains a unified api for training and inferencing of different classification methods like Adaboost, Gradient Boost, LightGBM, Random forests, and Histogram gradient boosting. It also has an option to ensemble three of these methods. Along with this main class, there is also a cascading classifier [here](./classifier/cascading_classifier.py), that is relevant for heirarchial classification like in Yoga-82.

The training and evaluation scripts can be found [here](./api). It has k fold cross validation, along with a visualisation script to visualise the inference on an entire video. 

### Instructions to run

To reproduce the results, make necessary modifications (dataset paths, method etc) in the corresponding config files, and run `main.py`, with the path to the config as an argument. An example command to run frame wise evaluation on the in house dataset is shown below.

~~~
python3 main.py --cfg configs/in_house/frame_wise.yaml
~~~

Some relevant config files are:

- Frame wise Evaluation on in house dataset [(yaml)](./configs/in_house/frame_wise.yaml)
- Subject wise Evaluation on in house dataset [(yaml)](./configs/in_house/subject_wise.yaml)
- Camera wise Evaluation, training on 3 camera angles of in house dataset [(yaml)](./configs/in_house/camera_wise_3cam.yaml)



---


This is being developed as part of the Yogasana classification project under Prof. Rahul Garg, CSE, IITD.
