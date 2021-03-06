"""
Creator: Ivanovitch Silva
Date: 14 Mar. 2022
Evaluate the inference model using test dataset and encoder artifact.
"""
# from train.transformer_feature import FeatureSelector, NumericalTransformer, CategoricalTransformer
import argparse
import logging
import pandas as pd
import joblib
import json
from collections import defaultdict
import sys
import pathlib
import os
# append the pipeline folder into the path
path = os.path.join(pathlib.Path.cwd(), "pipeline")
sys.path.append(path)
from train.helper import inference, compute_model_metrics
# print(sys.path)

# configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

# reference for a logging obj
logger = logging.getLogger()


def process_args(args):

    logger.info("Downloading and reading test artifact")
    df_test = pd.read_csv(args.test_data)

    # Extract the target from the features
    logger.info("Extracting target from dataframe")
    x_test = df_test.copy()
    y_test = x_test.pop("salary")

    # Extract the encoding of the target variable
    logger.info("Extracting the encoding of the target variable")
    le = joblib.load(args.encoder)

    # transform y_train
    y_test = le.transform(y_test)
    logger.info("Classes [0, 1]: {}".format(le.inverse_transform([0, 1])))

    # Download inference artifact
    logger.info("Downloading and load the exported model")
    pipe = joblib.load(args.model)

    # Predict test data
    predict = inference(pipe, x_test)

    print(type(predict))
    print(sum(predict))

    # Evaluation Metrics
    logger.info("Evaluation metrics")
    acc, precision, recall, fbeta = compute_model_metrics(y_test, predict)

    print(type(acc), type(precision), type(recall), type(fbeta))

    logger.info("Accuracy: {}".format(acc))
    logger.info("Precision: {}".format(precision))
    logger.info("Recall: {}".format(recall))
    logger.info("F1: {}".format(fbeta))

    # store all evaluation metrics
    with open(args.score_file, "w") as fd:
        json.dump({"accuracy": acc,
                   "precision": precision,
                   "recall": recall,
                   "f1": fbeta},
                  fd,
                  indent=4)

    # slicing performance
    score_dict = defaultdict(list)
    for feature in x_test.select_dtypes("object").columns:
        for cardinality in x_test[feature].unique():
            filter_slice = x_test[feature] == cardinality
            slice_rows = x_test[filter_slice]
            slice_target = y_test[filter_slice]
            predict = inference(pipe, slice_rows)
            acc, precision, recall, fbeta = compute_model_metrics(
                slice_target, predict)
            score_dict[feature].append({"slice": cardinality,
                                        "accuracy": acc,
                                        "precision": precision,
                                        "recall": recall,
                                        "f1": fbeta
                                        })
    # store all slices
    with open(args.slice_file, "w") as file:
        json.dump(score_dict, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the provided model on the test artifact",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Fully-qualified artifact name for the exported model to evaluate",
        required=True)

    parser.add_argument(
        "--test_data",
        type=str,
        help="Fully-qualified artifact name for the test data",
        required=True
    )

    parser.add_argument(
        "--encoder",
        type=str,
        help="Fully-qualified artifact name for the encoder used in the target variable",
        required=True)

    parser.add_argument(
        "--score_file",
        type=str,
        help="Json file used to store score results",
        required=True
    )

    parser.add_argument(
        "--slice_file",
        type=str,
        help="Json file used to store slice output",
        required=True
    )

    ARGS = parser.parse_args()

    process_args(ARGS)
