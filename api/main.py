"""
Creator: Ivanovitch Silva
Date: 26 Mar. 2022
Create API
"""
import pandas as pd
import joblib
import sys
import pathlib
import os
# append the folder into the path
path = os.path.join(pathlib.Path.cwd(), "")
sys.path.append(path)
from pipeline.train.helper import inference
# print(sys.path)

# import libraries to create and deploy API
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# Import Union since our Item object will have tags that can be strings or a list.
from typing import Union 
# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel

# create the api
app = FastAPI()

# declare request example data using pydantic
# a person in our dataset has the following attributes
class Person(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
                "example": {
                    "age": 72,
                    "workclass": 'Self-emp-inc',
                    "fnlwgt": 473748,
                    "education": 'Some-college',
                    "education_num": 10,
                    "marital_status": 'Married-civ-spouse',
                    "occupation": 'Exec-managerial',
                    "relationship": 'Husband',
                    "race": 'White',
                    "sex": 'Male',
                    "capital_gain": 0,
                    "capital_loss": 0,
                    "hours_per_week": 25,
                    "native_country": 'United-States'
                }
        }
        
# give a greeting using GET
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <p><span style="font-size:28px"><strong>Hello World</strong></span></p>"""\
    """<p><span style="font-size:20px">In this project, we will apply the skills """\
    """acquired in the Deploying a Scalable ML Pipeline in Production course to develop """\
    """a classification model on publicly available"""\
    """<a href="http://archive.ics.uci.edu/ml/datasets/Adult"> Census Bureau data</a>.</span></p>"""

# run the model inference and use a Person data structure via POST to the API.
@app.post("/predict")
async def get_inference(person: Person):
    # Download inference artifact
    pipe = joblib.load("pipeline/data/model_export")
        
    # Create a dataframe from the input feature
    # note that we could use pd.DataFrame.from_dict
    # but due be only one instance, it would be necessary to 
    # pass the Index.    
    df = pd.DataFrame([person.dict()])
    
    # Predict test data
    predict_value = inference(pipe, df)

    return "low income <=50K" if predict_value[0] <= 0.5 else "high income >50K"