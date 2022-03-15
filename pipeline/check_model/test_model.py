"""
Creator: Ivanovitch Silva
Date: 14 Mar. 2022
Implementing the test model functions. 
"""
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import sys
import pathlib
import os
# append the pipeline folder into the path
path = os.path.join(pathlib.Path.cwd(),"pipeline")
sys.path.append(path)
print(sys.path)
from train.helper import inference, compute_model_metrics
import numpy

# Encoder target type is correct?
def test_encoder_target(data):
    _, _, le = data

    assert isinstance(le, LabelEncoder)
    
# Model type is correct?
def test_model(data):
    _, pipe, _ = data

    assert isinstance(pipe, Pipeline)
    
# Inference return a correct type?
def test_inference(data):
    df_test, pipe, le = data
    
    # Extracting target from dataframe
    x_test = df_test.copy()
    _ = x_test.pop("salary")
    
    # Predict test data
    predict = inference(pipe, x_test)
    
    assert isinstance(predict, numpy.ndarray)
    