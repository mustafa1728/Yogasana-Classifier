import argparse
import logging
from config import get_cfg_defaults
import os

from api.kfold_cross_val import Kfold_cross_val


def get_args():
    parser = argparse.ArgumentParser(description="Yogasana Classification")
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    print(cfg)

    os.makedirs(cfg.OUTPUT.OUTPUT_DIR, exist_ok=True)
    save_model_path = os.path.join(cfg.OUTPUT.OUTPUT_DIR, cfg.OUTPUT.MODEL_NAME)  
    log_path = os.path.join(cfg.OUTPUT.OUTPUT_DIR, cfg.OUTPUT.LOG_FILE)  
    preds_path = os.path.join(cfg.OUTPUT.OUTPUT_DIR, cfg.OUTPUT.PREDICTIONS_NAME)  

    logging.basicConfig(filename=log_path, filemode='a', format='%(levelname)s | %(message)s', level=logging.INFO)
    logging.info(str(cfg))

    Kfold_cross_val(
        max_depth = cfg.SOLVER.MAX_DEPTH,
        no_trees = cfg.SOLVER.NO_TREES,
        n_splits = cfg.EVALUATION.N_SPLITS,
        dataset_path = cfg.DATASET.PATH,
        save_model_path = save_model_path,
        predictions_path = preds_path,
        method = cfg.SOLVER.METHOD,
        lr = cfg.SOLVER.LR,
        split_type = cfg.EVALUATION.METHOD,
        log_path = log_path,
        n_cams = cfg.EVALUATION.N_CAMS,
    )


if __name__ == "__main__":
    main()