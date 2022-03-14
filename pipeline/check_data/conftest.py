"""
Creator: Ivanovitch Silva
Date: 07 Mar. 2022
We can define the fixture functions in this file to make
them accessible across multiple test files.
"""
import pytest
import pandas as pd
import yaml
from yaml import CLoader as Loader
import os

def pytest_addoption(parser):
    parser.addoption("--sample_artifact", action="store")
    parser.addoption("--param", action="store")

@pytest.fixture(scope="session")
def data(request):
    
    # read the name of sample_artifact
    sample_artifact = request.config.option.sample_artifact
    
    if sample_artifact is None:
        pytest.fail("--sample_artifact missing on command line")
    
    # read sampe_artifact to a dataframe
    sample1 = pd.read_csv(sample_artifact)

    # read the name of the yaml file
    parameter_artifact = request.config.option.param
    
    if parameter_artifact is None:
        pytest.fail("--param missing on command line")
        
    # open and read the yaml file
    with open(parameter_artifact, "rb") as yaml_file:
        params = yaml.load(yaml_file, Loader=Loader)
    
    # read the reference dataset to a dataframe
    sample2 = pd.read_csv(params["data"]["reference_dataset"])
    
    # read the ks_alpha parameter
    ks_alpha = float(params["data"]["ks_alpha"])
    
    return sample1, sample2, ks_alpha