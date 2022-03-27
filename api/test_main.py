"""
Creator: Ivanovitch Silva
Date: 27 Mar. 2022
Test API
"""
from fastapi.testclient import TestClient
import os
import sys
import pathlib
# append the folder into the path
path = os.path.join(pathlib.Path.cwd(), "")
sys.path.append(path)
from api.main import app

# Instantiate the testing client with our app.
client = TestClient(app)

def test_root():
    r = client.get("/")
    assert r.status_code == 200

def test_get_inference_low_income():
    
    person = {
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
    
    r = client.post("/predict", json=person)
    print(r.json())
    assert r.status_code == 200
    assert r.json() == "low income <=50K"

def test_get_inference_high_income():
    
    person = {
        "age": 61,
        "workclass": 'Local-gov',
        "fnlwgt": 144723,
        "education": 'Masters',
        "education_num": 14,
        "marital_status": 'Married-civ-spouse',
        "occupation": 'Exec-managerial',
        "relationship": 'Husband',
        "race": 'White',
        "sex": 'Male',
        "capital_gain": 7298,
        "capital_loss": 0,
        "hours_per_week": 60,
        "native_country": 'United-States'
    }
    
    r = client.post("/predict", json=person)
    print(r.json())
    assert r.status_code == 200
    assert r.json() == "high income >50K"