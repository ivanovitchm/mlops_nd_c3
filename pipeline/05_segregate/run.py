"""
Creator: Ivanovitch Silva
Date: 07 Mar. 2022
Split preprocessed data into train and test.
"""
import argparse
import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
from yaml import CLoader as Loader

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
        args.input_artifact: Fully qualified name for the artifact
        args.param: Yaml file used to store configurable parameters
    """
    logger.info("Reading the preprocessed data")
    input_artifact = args.input_artifact

    df = pd.read_csv(input_artifact)
    
    # open and read the yaml file
    with open(args.param, "rb") as yaml_file:
        params = yaml.load(yaml_file, Loader=Loader)
    
    # read all parameters
    test_size = float(params["data"]["test_size"])
    random_state = int(params["main"]["random_seed"])
    stratify= params["data"]["stratify"]

    # Split first in model_dev/test, then we further divide model_dev in train and validation
    logger.info("Splitting data into train, val and test")
    splits = {}

    splits["train"], splits["test"] = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[stratify] if stratify != 'null' else None
    )

    # Save the output artifacts. 
    for split, df in splits.items():

        # Make the artifact name from the name of the split
        artifact_name = f"{split}_data.csv"

        # Get the path on disk 
        temp_path = os.path.join("pipeline/01_data", artifact_name)

        logger.info(f"Uploading the {split} dataset to {artifact_name}")

        # Save then upload to disk
        df.to_csv(temp_path,index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a dataset into train and test",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--param",
        type=str,
        help="Yaml file used to store configurable parameters",
        required=True
    )

    ARGS = parser.parse_args()

    process_args(ARGS)