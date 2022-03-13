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
import json 
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import fbeta_score, precision_score, recall_score
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

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
    
    
#Custom Transformer that extracts columns passed as argument to its constructor 
class FeatureSelector( BaseEstimator, TransformerMixin ):
    #Class Constructor 
    def __init__( self, feature_names ):
        self.feature_names = feature_names 
    
    #Return self nothing else to do here    
    def fit( self, X, y = None ):
        return self 
    
    #Method that describes what this custom transformer need to do
    def transform( self, X, y = None ):
        return X[ self.feature_names ]
        
# Handling categorical features 
class CategoricalTransformer( BaseEstimator, TransformerMixin ):
    # Class constructor method that takes one boolean as its argument
    def __init__(self, new_features=True, colnames=None):
        self.new_features = new_features
        self.colnames = colnames

    #Return self nothing else to do here    
    def fit( self, X, y = None ):
        return self 

    def get_feature_names(self):
        return self.colnames.tolist()

    # Transformer method we wrote for this transformer 
    def transform(self, X , y = None):
        df = pd.DataFrame(X,columns=self.colnames)

        # customize feature?
        # how can I identify this one? EDA!!!!
        if self.new_features: 

            # minimize the cardinality of native_country feature
            df.loc[df['native_country']!='United-States','native_country'] = 'non_usa' 

            # replace ? with Unknown
            edit_cols = ['native_country','occupation','workclass']
            for col in edit_cols:
                df.loc[df[col] == '?', col] = 'unknown'

            # decrease the cardinality of education feature
            hs_grad = ['HS-grad','11th','10th','9th','12th']
            elementary = ['1st-4th','5th-6th','7th-8th']
            # replace
            df['education'].replace(to_replace = hs_grad,value = 'HS-grad',inplace = True)
            df['education'].replace(to_replace = elementary,value = 'elementary_school',inplace = True)
            
            # adjust marital_status feature
            married= ['Married-spouse-absent','Married-civ-spouse','Married-AF-spouse']
            separated = ['Separated','Divorced']
            
            # replace 
            df['marital_status'].replace(to_replace = married ,value = 'Married',inplace = True)
            df['marital_status'].replace(to_replace = separated,value = 'Separated',inplace = True)

            # adjust workclass feature
            self_employed = ['Self-emp-not-inc','Self-emp-inc']
            govt_employees = ['Local-gov','State-gov','Federal-gov']
            
            # replace elements in list.
            df['workclass'].replace(to_replace = self_employed ,value = 'Self_employed',inplace = True)
            df['workclass'].replace(to_replace = govt_employees,value = 'Govt_employees',inplace = True)

        # update column names
        self.colnames = df.columns      

        return df

# transform numerical features
class NumericalTransformer( BaseEstimator, TransformerMixin ):
    # Class constructor method that takes a model parameter as its argument
    # model 0: minmax
    # model 1: standard
    # model 2: without scaler
    def __init__(self, model = 0, colnames=None):
        self.model = model
        self.colnames = colnames

    #Return self nothing else to do here    
    def fit( self, X, y = None ):
        return self

    # return columns names after transformation
    def get_feature_names(self):
        return self.colnames 

    #Transformer method we wrote for this transformer 
    def transform(self, X , y = None ):
        df = pd.DataFrame(X,columns=self.colnames)

        # update columns name
        self.colnames = df.columns.tolist()

        # minmax
        if self.model == 0: 
            scaler = MinMaxScaler()
            # transform data
            df = scaler.fit_transform(df)
        elif self.model == 1:
            scaler = StandardScaler()
            # transform data
            df = scaler.fit_transform(df)
        else:
            df = df.values

        return df

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