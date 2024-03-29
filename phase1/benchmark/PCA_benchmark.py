import sys
import logging
from pyod.models.pca import PCA

sys.path.append("../")

from common import data_preprocess
from common.dataloader import load_dataset
from common.utils import seed_everything, load_config, set_logger, print_to_json
from common.evaluation import Evaluator, TimeTracker
from common.exp import store_entity

seed_everything()
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./benchmark_config/",
        help="The config directory.",
    )
    parser.add_argument("--expid", type=str, default="pca_SMD")
    parser.add_argument("--gpu", type=int, default=-1)
    args = vars(parser.parse_args())

    config_dir = args["config"]
    experiment_id = args["expid"]

    params = load_config(config_dir, experiment_id)
    set_logger(params, args)
    logging.info(print_to_json(params))

    data_dict = load_dataset(
        data_root=params["data_root"],
        entities=params["entities"],
        dim=params["dim"],
        valid_ratio=params["valid_ratio"],
        test_label_postfix=params["test_label_postfix"],
        test_postfix=params["test_postfix"],
        train_postfix=params["train_postfix"],
        nrows=params["nrows"],
    )

    # preprocessing
    pp = data_preprocess.preprocessor(model_root=params["model_root"])
    data_dict = pp.normalize(data_dict, method=params["normalize"])

    # train/test on each entity put here
    evaluator = Evaluator(**params["eval"])
    for entity in params["entities"]:
        logging.info("Fitting dataset: {}".format(entity))
        train = data_dict[entity]["train"]
        test = data_dict[entity]["test"]
        test_label = data_dict[entity]["test_label"]

        # data preprocessing for MSCRED
        model = PCA()

        tt = TimeTracker()
        tt.train_start()
        model.fit(train)
        tt.train_end()

        # get outlier scores
        train_anomaly_score = model.decision_function(train)
        tt.test_start()
        anomaly_score = model.decision_function(test)
        tt.test_end()
        anomaly_label = test_label

        if isinstance(anomaly_score, tuple):  # tuple check
            anomaly_score = anomaly_score[0]

            if len(anomaly_score.shape) == 2:  # 2d 1d check
                anomaly_score = anomaly_score.flatten()

        store_entity(
            params,
            entity,
            train_anomaly_score,
            anomaly_score,
            anomaly_label,
            time_tracker=tt.get_data(),
        )
    evaluator.eval_exp(
        exp_folder=params["model_root"],
        entities=params["entities"],
        merge_folder=params["benchmark_dir"],
        extra_params=params,
    )
