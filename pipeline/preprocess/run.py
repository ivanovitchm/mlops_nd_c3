"""
Creator: Ivanovitch Silva
Date: 05 Mar. 2022
After fetching the raw data we need to preprocessing it.
At the end of this stage we will have created a new artifact (preprocessed_data.csv).
"""
import argparse
import logging
import pandas as pd

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
        args.input_artifact_name: Fully qualified name for the raw data artifact
        args.output_artifact_name: Name for the clean data artifact that will be created
    """

    logger.info("Fetching input data artifact")
    input_artifact = args.input_artifact_name

    # columns used
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
               'marital_status', 'occupation', 'relationship', 'race',
               'sex', 'capital_gain', 'capital_loss', 'hours_per_week',
               'native_country', 'salary']

    # create a dataframe from the artifact path
    df = pd.read_csv(input_artifact)
    df.columns = columns

    # Delete duplicated rows
    logger.info("Dropping duplicates")
    df.drop_duplicates(inplace=True)

    # Remove white space in categorical features
    # Tip: we need guarantee this procedure be also adopted during production
    # stage
    logger.info("Remove white space in categorical features")
    df_cat = df.select_dtypes(["object"])
    df[df_cat.columns] = df_cat.apply(lambda row: row.str.strip())

    # Generate the clean data artifact. e.g path/preprocessed_data.csv
    logger.info("Generate the clean data artifact")
    out_artifact = args.output_artifact_name
    df.to_csv(out_artifact, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess a dataset",
        fromfile_prefix_chars="@"
    )

    parser.add_argument(
        "--input_artifact_name",
        type=str,
        help="Fully-qualified name for the raw input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact_name",
        type=str,
        help="Name for the clean data artifact",
        required=True
    )

    # get arguments
    ARGS = parser.parse_args()

    # process the arguments
    process_args(ARGS)
