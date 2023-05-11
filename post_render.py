import requests
import json


url = "https://fastapi-for-udacity-mlops-project.onrender.com/inference"


# explicit the sample to perform inference on
sample =  { 'age':50,
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


# post to API and collect response
response = requests.post(url, json=sample )

# display output - response will show sample details + model prediction added
print("response status code", response.status_code)
print("response content:")
print(response.json())