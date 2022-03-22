"""
Creator: Ivanovitch Silva
Date: 14 Mar. 2022
Functions used to create the pipeline, save artifacts,
computer metrics and perform inference
"""
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import fbeta_score, precision_score, recall_score, accuracy_score
import joblib
import os
import pathlib
import sys
# append the pipeline folder into the path
path = os.path.join(pathlib.Path.cwd(), "pipeline")
sys.path.append(path)
from train.transformer_feature import FeatureSelector, CategoricalTransformer, NumericalTransformer


def generate_pipeline(x_train, numerical_model, model_config):
    """Generate the data pipeline
    Arguments
    x_train: independent features which need to be transformed
    numerical_model: parameter of the numerical transformer
    model_config: configurations of the model

    Return
    pipeline object used to process all features
    """

    # Categrical features to pass down the categorical pipeline
    categorical_features = x_train.select_dtypes("object").columns.to_list()

    # Numerical features to pass down the numerical pipeline
    numerical_features = x_train.select_dtypes("int64").columns.to_list()

    # Defining the steps in the categorical pipeline
    categorical_pipeline = Pipeline(steps=[('cat_selector', FeatureSelector(categorical_features)),
                                           ('imputer_cat', SimpleImputer(
                                               strategy="most_frequent")),
                                           ('cat_transformer', CategoricalTransformer(
                                               colnames=categorical_features)),
                                           # ('cat_encoder','passthrough'
                                           ('cat_encoder', OneHotEncoder(
                                               sparse=False, drop="first"))
                                           ]
                                    )

    # Defining the steps in the numerical pipeline
    numerical_pipeline = Pipeline(
        steps=[
            ('num_selector', FeatureSelector(numerical_features)), ('imputer_num', SimpleImputer(
                strategy="median")), ('num_transformer', NumericalTransformer(
                    numerical_model, colnames=numerical_features))])

    # Combining numerical and categorical piepline into one full big pipeline horizontally
    # using FeatureUnion
    full_pipeline_preprocessing = FeatureUnion(
        transformer_list=[
            ('cat_pipeline', categorical_pipeline), ('num_pipeline', numerical_pipeline)])

    # The full pipeline
    pipe = Pipeline(
        steps=[
            ('full_pipeline',
             full_pipeline_preprocessing),
            ("classifier",
             DecisionTreeClassifier(
                 **model_config))])
    return pipe


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


def save_artifact(pipe, pipeline_path, encoder, encoder_path):
    """Save the artifacts

    Arguments
    pipe: trained data pipeline
    pipeline_path: fully qualified artifact name of pipeline
    encoder: target encoder
    encoder_path: fully qualified artifact name of the enconder
    """

    # Export if required
    if pipeline_path != "null":
        # Save the model using joblib
        joblib.dump(pipe, pipeline_path)

    if encoder_path != "null":
        # Save the target encoder using joblib
        joblib.dump(encoder, encoder_path)
