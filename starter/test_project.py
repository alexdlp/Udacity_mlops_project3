
import logging
import pandas as pd
import os


def test_data_shape(data):
    """
    Test shape of the data
    """
    # Check the df shape
    try:
        assert data.shape[0] > 0
        assert data.shape[1] > 0

    except AssertionError as err:
        logging.error(
        "Testing dataset: The file doesn't appear to have rows and columns")
        raise err

def test_data_features(data, features):
    """
    Test features of the data
    """
    try:

        assert set(data.columns) == set(features)
    
    except AssertionError as err:
        logging.error(
        "Testing dataset: Features are missing in the data columns")
        raise err


def test_model(model, dataset_split):
    """
    Check if model is able to make predictions
    """
    try:
        X_train, y_train, X_test, y_test = dataset_split
        preds = model.predict(X_test)
        assert preds.shape[0] == X_test.shape[0]

    except Exception as err:
        logging.error(
        "Testing model: Saved model is not able to make new predictions")
        raise err

from fastapi.testclient import TestClient
#from fastapi import HTTPException
import json
import logging
from main import app

client = TestClient(app)

def test_get():
    """
    Test welcome message for get at root
    """
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == "The API is working!"



def test_post_50k(data):

    # take a sample from the data with a high salary
    sample=data[data['salary']=='<=50K'].iloc[500].to_dict()
    # remove target
    _ = sample.pop('salary')

    data = json.dumps(sample)

    r = client.post("/inference/", data=data )

     # test response and output
    assert r.status_code == 200
    assert r.json()["prediction"] == '<=50K'

def test_post_0k(data):

    # take a sample from the data with a high salary
    sample=data[data['salary']=='>50K'].iloc[500].to_dict()
    # remove target
    _ = sample.pop('salary')

    data = json.dumps(sample)

    r = client.post("/inference/", data=data )

     # test response and output
    assert r.status_code == 200
    assert r.json()["prediction"] == '>50K'




