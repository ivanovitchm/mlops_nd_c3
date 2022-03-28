"""
Creator: Ivanovitch Silva
Date: 26 Mar. 2022
Script that POSTS to the API using the requests 
module and returns both the result of 
model inference and the status code
"""
import requests
import json
import pprint

person = {
        "age": 46,
        "workclass": 'Private',
        "fnlwgt": 364548,
        "education": 'Bachelors',
        "education_num": 13,
        "marital_status": 'Divorced',
        "occupation": 'Sales',
        "relationship": 'Not-in-family',
        "race": 'White',
        "sex": 'Male',
        "capital_gain": 8614,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": 'United-States'
    }

url = "https://income-census-project.herokuapp.com"
response = requests.post(f"{url}/predict",
                         json=person)

print("Request: https://income-census-project.herokuapp.com/predict")
print(f"Person: \n age: {person['age']}\n workclass: {person['workclass']}\n"\
      f" fnlwgt: {person['age']}\n education: {person['education']}\n"\
      f" education_num: {person['education_num']}\n"\
      f" marital_status: {person['marital_status']}\n"\
      f" occupation: {person['occupation']}\n"\
      f" relationship: {person['relationship']}\n"\
      f" race: {person['race']}\n"\
      f" sex: {person['sex']}\n"\
      f" capital_gain: {person['capital_gain']}\n"\
      f" capital_loss: {person['capital_loss']}\n"\
      f" hours_per_week: {person['hours_per_week']}\n"\
      f" native_country: {person['native_country']}\n"
     )
print(f"Result of model inference: {response.json()}")
print(f"Status code: {response.status_code}")