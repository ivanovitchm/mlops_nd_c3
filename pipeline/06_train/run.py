"""
Creator: Ivanovitch Silva
Date: 08 Mar. 2022
Implement a machine pipeline component that
incorporate preprocessing and train stages.
"""
import argparse
import logging
import yaml
from yaml import CLoader as Loader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder
from helper import generate_pipeline, inference, compute_model_metrics, save_artifact
import json 

# configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

# reference for a logging obj
logger = logging.getLogger()

def process_args(args):
    """
    Arguments
        args - command line arguments
        args.train_data: Fully-qualified name for the training data artifact
        args.param: Yaml file used to store configurable parameters
        args.score_file: Json file used to store the score results
    """

    logger.info("Downloading and reading train artifact")
    local_path = args.train_data
    df_train = pd.read_csv(local_path)
    
    # open and read the yaml file
    with open(args.param, "rb") as yaml_file:
        params = yaml.load(yaml_file, Loader=Loader)
    
    # read all parameters
    val_size = float(params["data"]["val_size"])
    random_state = int(params["main"]["random_seed"])
    stratify = params["data"]["stratify"]
    export_artifact = params["train"]["export_artifact"]
    numerical_model = params["train"]["numerical_pipe"]["model"]
    decision_tree_config = params["train"]["decision_tree"]
    export_encoder = params["train"]["export_encoder"]

    # Spliting train.csv into train and validation dataset
    logger.info("Spliting data into train/val")
    # split-out train/validation and test dataset
    x_train, x_val, y_train, y_val = train_test_split(df_train.drop(labels=stratify,axis=1),
                                                      df_train[stratify],
                                                      test_size=val_size,
                                                      random_state=random_state,
                                                      shuffle=True,
                                                      stratify=df_train[stratify])
    
    logger.info("x train: {}".format(x_train.shape))
    logger.info("y train: {}".format(y_train.shape))
    logger.info("x val: {}".format(x_val.shape))
    logger.info("y val: {}".format(y_val.shape))

    logger.info("Removal Outliers")
    # temporary variable
    x = x_train.select_dtypes("int64").copy()

    # identify outlier in the dataset
    lof = LocalOutlierFactor()
    outlier = lof.fit_predict(x)
    mask = outlier != -1

    logger.info("x_train shape [original]: {}".format(x_train.shape))
    logger.info("x_train shape [outlier removal]: {}".format(x_train.loc[mask,:].shape))

    # dataset without outlier, note this step could be done during the preprocesing stage
    x_train = x_train.loc[mask,:].copy()
    y_train = y_train[mask].copy()

    logger.info("Encoding Target Variable")
    # define a categorical encoding for target variable
    le = LabelEncoder()

    # fit and transform y_train
    y_train = le.fit_transform(y_train)

    # transform y_test (avoiding data leakage)
    y_val = le.transform(y_val)
    
    logger.info("Classes [0, 1]: {}".format(le.inverse_transform([0, 1])))
    
    # Pipeline generation
    logger.info("Pipeline generation")
    pipe = generate_pipeline(x_train, numerical_model,decision_tree_config)

    # training 
    logger.info("Training")
    pipe.fit(x_train,y_train)

    # predict
    logger.info("Infering")
    predict = inference(pipe, x_val)
    
    # Evaluation Metrics
    logger.info("Evaluation metrics")
    acc, precision, recall, fbeta = compute_model_metrics(y_val, predict)
    
    logger.info("Accuracy: {}".format(acc))
    logger.info("Precision: {}".format(precision))
    logger.info("Recall: {}".format(recall))
    logger.info("F1: {}".format(fbeta))
     
    with open(args.score_file, "w") as fd:
        json.dump({"accuracy": acc,
                   "precision": precision,
                   "recall": recall,
                   "f1": fbeta},
                  fd,
                  indent=4)

    # save artifacts to disk
    save_artifact(pipe, export_artifact, le, export_encoder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Decision Tree",
        fromfile_prefix_chars="@",
    )
    
    parser.add_argument(
        "--train_data",
        type=str,
        help="Fully-qualified name for the training data artifact",
        required=True,
    )
    
    parser.add_argument(
        "--param",
        type=str,
        help="Yaml file used to store configurable parameters",
        required=True
    )
    
    parser.add_argument(
        "--score_file",
        type=str,
        help="Json file used to store score results",
        required=True
    )

    ARGS = parser.parse_args()

    process_args(ARGS)