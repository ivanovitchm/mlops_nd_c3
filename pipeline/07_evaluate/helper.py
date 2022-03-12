"""
Creator: Ivanovitch Silva
Date: 12 Mar. 2022
Functions used to help the test evaluation
"""
from sklearn.metrics import fbeta_score, precision_score, recall_score

def inference(pipe, x):
    """ Run model inferences and return the predictions.
    
    Arguments
    pipeline: data pipeline
    x : features
    
    Returns
        Predictions from the model.
    """
    return pipe.predict(x)

def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.
    
    Arguments
    y : Known labels, binarized.
    preds : Predicted labels, binarized.
    
    Returns
    The follow metrics: accuracy, precision, recall and fbeta
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    acc = accuracy_score(y, preds)
    
    return acc, precision, recall, fbeta