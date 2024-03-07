from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel
from io import StringIO
import pandas as pd
import numpy as np
import requests
import mlflow
import pickle
import json
import os



from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from autogluon.tabular import TabularDataset, TabularPredictor

#




model_dir="FinalModel-SoilMoisture"

# Get the absolute path
absolute_path = os.path.abspath(model_dir)

# Load the AutoGluon model
model = TabularPredictor.load(path=str(absolute_path), require_version_match=False, require_py_version_match=False)

# To access endpoint / FastAPI UI, add /docs#/ to endpoint
# e.g. http://127.0.0.1:8000/docs#/


class Input(BaseModel):
    atm_pressure_kPa : float
    precipitation_mm : float
    Soil_conductivity_5cm_S000988 : float
    radiation_W_m2 : float
    rel_humidity : float
    Temp_2m_Celsius : float
    windspeed_m_s : float



app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the Soil Moisture Prediction App"}


@app.put('/predict')
def predict(d:Input):
    # Preprocess the model inputs
    d.atm_pressure_kPa*=1
    d.precipitation_mm*=1
    d.Soil_conductivity_5cm_S000988*=1
    d.radiation_W_m2*=1
    d.rel_humidity*=1
    d.Temp_2m_Celsius*=1
    d.windspeed_m_s*=1
    
    # Convert inputs into a pandas dataframe
    X_inputs={"atm_pressure_kPa":[d.atm_pressure_kPa], 
           "precipitation_mm":[d.precipitation_mm], 
           "Soil_conductivity_5cm_S000988":[d.Soil_conductivity_5cm_S000988], 
           "radiation_W_m2":[d.radiation_W_m2], 
           "rel_humidity":[d.rel_humidity],
           "Temp_2m_Celsius":[d.Temp_2m_Celsius],
           "windspeed_m_s":[d.windspeed_m_s]}
    
    X_inputs_df = pd.DataFrame(X_inputs)

    # Run inference using model inputs
    #soil_moisture_5cm_S000988 = model.predict(X_inputs_df, model='WeightedEnsemble_L2')  
    
    #return the numeric portion of the model prediction
    #return soil_moisture_5cm_S000988[0] 
    return d.atm_pressure_kPa*d.Soil_conductivity_5cm_S000988*100


@app.get('/predict')
#def predict(atm_pressure_kPa:float, precipitation_mm:float, Soil_conductivity_5cm_S000988:float, 
#            radiation_W_m2:float, rel_humidity:float, Temp_2m_Celsius:float, windspeed_m_s:float): 
def predict(d:Input):       
    # Preprocess the model inputs
    d.atm_pressure_kPa*=1.0
    d.precipitation_mm*=1.0 
    d.Soil_conductivity_5cm_S000988*=1.0
    d.radiation_W_m2*=1.0
    d.rel_humidity*=1.0
    d.Temp_2m_Celsius*=1.0
    d.windspeed_m_s*=1.0
    
    # Convert inputs into a pandas dataframe
    X_inputs={"atm_pressure_kPa":[d.atm_pressure_kPa], 
           "precipitation_mm":[d.precipitation_mm], 
           "Soil_conductivity_5cm_S000988":[d.Soil_conductivity_5cm_S000988], 
           "radiation_W_m2":[d.radiation_W_m2], 
           "rel_humidity":[d.rel_humidity],
           "Temp_2m_Celsius":[d.Temp_2m_Celsius],
           "windspeed_m_s":[d.windspeed_m_s]}
    
    X_inputs_df = pd.DataFrame(X_inputs)


    # Run inference using model inputs
    #soil_moisture_5cm_S000988 = model.predict(X_inputs_df, model='WeightedEnsemble_L2')  
    
    #return the numeric portion of the model prediction
    #return soil_moisture_5cm_S000988[0] 
    return d.atm_pressure_kPa*d.Soil_conductivity_5cm_S000988*100



@app.post('/predict')
async def single_inference(X_inputs: dict):
    url = 'http://127.0.0.1:7070/invocations'
    headers={'Content-Type':'application/json'}
    X_data = X_inputs.dict()
    # model input must be list of lists or pandas dataframe
    data_in=[[X_data['age'], X_data['sex'],X_data['bmi'],X_data['bp'],X_data['s1'],
             X_data['s2'],X_data['s3'],X_data['s4'],X_data['s5'],X_data['s6']]]
    print(data_in)
    #inference_request=json.dumps({'inputs':[inputs]})
    inference_request={"inputs":data_in}
    print(inference_request)
    #response=requests.post(url,inference_request,headers=headers)
    response=requests.post(url,json=inference_request)
    print(response)
    return response.text





@app.post('/predicts')
async def single_inferences(X_inputs: dict):
    #X_inputs=diabetesClass(**X_inputs)
    url = 'http://127.0.0.1:7070/invocations'
    headers = {'Content-Type': 'application/json'}
    X_data = np.array([X_inputs.values()])
    # model input must be list of lists or pandas dataframe
    #data_in = np.array([[X_data['age'], X_data['sex'], X_data['bmi'], X_data['bp'], X_data['s1'],
    #            X_data['s2'], X_data['s3'], X_data['s4'], X_data['s5'], X_data['s6']]])
    inference_request = {"inputs": X_data}
    print(inference_request)
    response = requests.post(url, json=inference_request, headers=headers)
    print(inference_request)
    return response.text


@app.post('/batch/')
async def batch_inference(file: bytes=File(...)):
    # Upload file to test batch predictions
    s=str(file,'utf-8')
    file=StringIO(s)
    #if file.filename.endswith('.csv'):
    #    data_df = pd.read_csv(file.file)
    if file.filename.endswith(('.csv', '.txt')):
        data_df = pd.read_csv(file.file, delimiter='\t')  # Use delimiter='\t' for tab-separated files
    elif file.filename.endswith(('.xls', '.xlsx')):
        data_df = pd.read_excel(file.file)
    elif file.filename.endswith('.json'):
        data_df = pd.read_json(file.file)
    else:
        return 'Unsupported file format. Supported formats are .csv, .xls, .xlsx and .json'

    url = 'http://127.0.0.1:7070/invocations'
    headers = {'Content-Type': 'application/json'}
    # model input must be list of lists or pandas dataframe
    #request_data = data_df.to_json(orient='values')
    inference_request = json.dumps({'inputs':data_df})
    response = requests.post(url, inference_request, headers=headers)
    print(inference_request)
    return response.text









    