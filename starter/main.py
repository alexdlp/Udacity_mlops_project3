# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel

from ml_model.ml.model import load_model, inference
from ml_model.ml.data import process_data
import os
import pandas as pd


MODEL_PATH = "./ml_model/model/"
class InputData(BaseModel):
    age: int
    workclass: str 
    fnlgt: int
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
                                    'age':50,
                                    'workclass':"Private", 
                                    'fnlgt':234721,
                                    'education':"Doctorate",
                                    'education_num':16,
                                    'marital_status':"Separated",
                                    'occupation':"Exec-managerial",
                                    'relationship':"Not-in-family",
                                    'race':"Black",
                                    'sex':"Female",
                                    'capital_gain':0,
                                    'capital_loss':0,
                                    'hours_per_week':50,
                                    'native_country':"United-States"
                                    }
                        }



# Instantiate the app.
app = FastAPI()

# load model artifacts on startup of the application to reduce latency
@app.on_event("startup")
async def startup_event(): 
    global model, encoder, lb
    # if the model exists, load the model
    if os.path.isfile(os.path.join(MODEL_PATH,'model.pkl')):       
        model_path  = os.path.join(MODEL_PATH,'model.pkl')
        encoder_path = os.path.join(MODEL_PATH,'encoder.pkl')
        labeler_path = os.path.join(MODEL_PATH,'labeler.pkl')
        model, encoder, lb = load_model(model_path, encoder_path, labeler_path)
        

# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return "The API is working!"


# This allows sending of data (our InferenceSample) via POST to the API.
@app.post("/inference/")
async def inference(inference: InputData):
    data = {  'age': inference.age,
                'workclass': inference.workclass, 
                'fnlgt': inference.fnlgt,
                'education': inference.education,
                'education-num': inference.education_num,
                'marital-status': inference.marital_status,
                'occupation': inference.occupation,
                'relationship': inference.relationship,
                'race': inference.race,
                'sex': inference.sex,
                'capital-gain': inference.capital_gain,
                'capital-loss': inference.capital_loss,
                'hours-per-week': inference.hours_per_week,
                'native-country': inference.native_country,
                }

    # prepare the sample for inference as a dataframe
    sample = pd.DataFrame(data, index=[0])
    print(sample)

    # apply transformation to sample data
    cat_features = [
                    "workclass",
                    "education",
                    "marital-status",
                    "occupation",
                    "relationship",
                    "race",
                    "sex",
                    "native-country",
                    ]

 
    sample,_,_,_ = process_data(
                                sample, 
                                categorical_features=cat_features, 
                                training=False, 
                                encoder=encoder, 
                                lb=lb
                                )

                         
    #prediction = inference(model,sample)
    prediction = model.predict(sample)

    # convert prediction to label and add to data output
    if prediction[0]>0.5:
        prediction = '>50K'
    else:
        prediction = '<=50K', 
    data['prediction'] = prediction


    return data

