"""
Creator: Ivanovitch Silva
Date: 07 Mar. 2022
We can define the fixture functions in this file to make
them accessible across multiple test files.
"""
import pytest
import pandas as pd

def pytest_addoption(parser):
    parser.addoption("--reference_artifact", action="store")
    parser.addoption("--sample_artifact", action="store")
    parser.addoption("--ks_alpha", action="store")

@pytest.fixture(scope="session")
def data(request):
    reference_artifact = request.config.option.reference_artifact

    if reference_artifact is None:
        pytest.fail("--reference_artifact missing on command line")

    sample_artifact = request.config.option.sample_artifact

    if sample_artifact is None:
        pytest.fail("--sample_artifact missing on command line")

    sample1 = pd.read_csv(reference_artifact)
    sample2 = pd.read_csv(sample_artifact)

    return sample1, sample2


@pytest.fixture(scope='session')
def ks_alpha(request):
    ks_alpha = request.config.option.ks_alpha

    if ks_alpha is None:
        pytest.fail("--ks_threshold missing on command line")

    return float(ks_alpha)