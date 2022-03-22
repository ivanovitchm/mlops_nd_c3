"""
Creator: Ivanovitch Silva
Date: 14 Mar. 2022
We can define the fixture functions in this file to make
them accessible across multiple test files.
"""
# from train.transformer_feature import FeatureSelector, NumericalTransformer, CategoricalTransformer
import pytest
import pandas as pd
import joblib
import sys
import pathlib
import os
# append the pipeline folder into the path
path = os.path.join(pathlib.Path.cwd(), "pipeline")
sys.path.append(path)
print(sys.path)


def pytest_addoption(parser):
    parser.addoption("--test_data", action="store")
    parser.addoption("--model", action="store")
    parser.addoption("--encoder", action="store")


@pytest.fixture(scope="session")
def data(request):

    # read the name of test_data
    test_data = request.config.option.test_data

    if test_data is None:
        pytest.fail("--test_data missing on command line")

    # Downloading and reading test artifact
    df_test = pd.read_csv(test_data)

    # read the model artifact
    model = request.config.option.model

    if model is None:
        pytest.fail("--model missing on command line")

    # Downloading and load the exported model
    pipe = joblib.load(model)

    # read the model artifact
    encoder = request.config.option.encoder

    if encoder is None:
        pytest.fail("--encoder missing on command line")

    # Extracting the encoding of the target variable
    le = joblib.load(encoder)

    return df_test, pipe, le
