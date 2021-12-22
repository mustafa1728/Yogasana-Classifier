# Yogasana-Classifier

Official codebase for the paper "A View Independent Classification Framework for Yoga Postures", under review at [ICVGIP 2021](https://iitj.ac.in/icvgip2021/).

### Installation

Packages required are listed in [requirements](./requirements.txt). There is a dependency on LighGBM ([paper](https://ui.adsabs.harvard.edu/abs/2021arXiv210900724Y/abstract)) ([docs](https://lightgbm.readthedocs.io/en/latest/)), and rest of the packages are standard libraries.

### Code Overview

Preprocessing codes to generate key points dataset from raw video data and alhpapose key points can be found [here](./preprocess). It also includes code for bounding box normalisation, key point selection and fold generation. 

The main classifier class can be found [here](./classifier/model.py). It contains a unified api for training and inferencing of different classification methods like Adaboost, Gradient Boost, LightGBM, Random forests, and Histogram gradient boosting. It also has an option to ensemble three of these methods. 

The training and evaluation scripts can be found [here](./api). It has k fold cross validation, along with a visualisation script to visualise the inference on an entire video. 

### Instructions to run

---


This is being developed as part of the Yogasana classification project under Prof. Rahul Garg, CSE, IITD.