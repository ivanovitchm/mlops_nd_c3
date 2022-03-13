"""
Creator: Ivanovitch Silva
Date: 12 Mar. 2022
Functions used to help the test evaluation
"""
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

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