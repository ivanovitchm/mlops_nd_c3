"""
Creator: Ivanovitch Silva
Date: 08 Mar. 2022
Implement a machine pipeline component that
incorporate preprocessing and train stages.



Refactoring

Class going to a separate file
Create a function to create pipeline
Create function to obtaion evaluaton metrics
Function to train()?
Try to encapsule the Y in the pipeline


"""
import argparse
import logging
import yaml
from yaml import CLoader as Loader
import joblib

from transformer_feature import FeatureSelector, CategoricalTransformer, NumericalTransformer

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer

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
    stratify= params["data"]["stratify"]
    export_artifact= params["train"]["export_artifact"]
    numerical_model= params["train"]["numerical_pipe"]["model"]
    decision_tree_config = params["train"]["decision_tree"]

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
    predict = pipe.predict(x_val)
    
    # Evaluation Metrics
    logger.info("Evaluation metrics")
    # Metric: AUC
    auc = roc_auc_score(y_val, predict, average="macro")
    
    # Metric: Accuracy
    acc = accuracy_score(y_val, predict)

    logger.info("AUC: {}".format(auc))
    logger.info("ACC: {}".format(acc))
    
    # Metric: Confusion Matrix
    # fig_confusion_matrix, ax = plt.subplots(1,1,figsize=(7,4))
    # ConfusionMatrixDisplay(confusion_matrix(predict,
    #                                         y_val,
    #                                         labels=[1,0]),
    #                        display_labels=[">50k","<=50k"]
    #                       ).plot(values_format=".0f",ax=ax)
    # ax.set_xlabel("True Label")
    # ax.set_ylabel("Predicted Label")
    
    # Export if required
    if export_artifact != "null":
        # Save the model using joblib
        joblib.dump(pipe, export_artifact)

def generate_pipeline(x_train, numerical_model, model_config):
    """
    Arguments
    x_train: independent features which need to be transformed
    numerical_model: parameter of the numerical transformer
    model_config: configurations of the model
    """

    # Categrical features to pass down the categorical pipeline 
    categorical_features = x_train.select_dtypes("object").columns.to_list()

    # Numerical features to pass down the numerical pipeline 
    numerical_features = x_train.select_dtypes("int64").columns.to_list()

    # Defining the steps in the categorical pipeline 
    categorical_pipeline = Pipeline(steps = [('cat_selector',FeatureSelector(categorical_features)),
                                             ('imputer_cat', SimpleImputer(strategy="most_frequent")),
                                             ('cat_transformer', CategoricalTransformer(colnames=categorical_features)),
                                             #('cat_encoder','passthrough'
                                             ('cat_encoder',OneHotEncoder(sparse=False,drop="first"))
                                            ]
                                   )
    
    # Defining the steps in the numerical pipeline     
    numerical_pipeline = Pipeline(steps = [('num_selector', FeatureSelector(numerical_features)),
                                           ('imputer_num', SimpleImputer(strategy="median")),
                                           ('num_transformer', NumericalTransformer(numerical_model,
                                                                                   colnames=numerical_features))
                                          ]
                                 )

    # Combining numerical and categorical piepline into one full big pipeline horizontally 
    # using FeatureUnion
    full_pipeline_preprocessing = FeatureUnion(transformer_list = [('cat_pipeline', categorical_pipeline),
                                                                   ('num_pipeline', numerical_pipeline)
                                                                  ]
                                              )

    # The full pipeline 
    pipe = Pipeline(steps = [('full_pipeline', full_pipeline_preprocessing),
                             ("classifier", DecisionTreeClassifier(**model_config))
                            ]
                   )
    return pipe
        
    
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

    ARGS = parser.parse_args()

    process_args(ARGS)