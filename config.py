"""
Default configurations for Yoga classification
"""

import os
from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.NAME = "in_house" # options: ["in_house", "yadav", "jain", "yoga_82"]
_C.DATASET.PATH = "dataset.csv"
# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.METHOD = "random_forest" # options: ["adaboost", "random_forest", "bagging", "grad_boost", "hist_grad_boost", "lgbm", "ensemble"]
_C.SOLVER.LR = 5
_C.SOLVER.MAX_DEPTH = 20
_C.SOLVER.NO_TREES = 500

# ---------------------------------------------------------------------------- #
# Output options
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CN()
_C.OUTPUT.OUTPUT_DIR = "output/frame_wise"
_C.OUTPUT.MODEL_NAME = "model.z"
_C.OUTPUT.PREDICTIONS_NAME = "predictions.csv"
_C.OUTPUT.LOG_FILE = "eval.log"
_C.OUTPUT.CONFUSION_ROOT = "confusion"

# ---------------------------------------------------------------------------- #
# EVALUATION options
# ---------------------------------------------------------------------------- #
_C.EVALUATION = CN()
_C.EVALUATION.METHOD = "frame" # options: ["frame", "subject", "camera"]
_C.EVALUATION.N_SPLITS = 10 



def get_cfg_defaults():
    return _C.clone()