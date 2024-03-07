from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel
from io import StringIO, BytesIO
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
    soil_moisture_5cm_S000988 = model.predict(X_inputs_df, model='WeightedEnsemble_L2')  
    
    #return the numeric portion of the model prediction
    return soil_moisture_5cm_S000988[0] 
    #return d.atm_pressure_kPa*d.Soil_conductivity_5cm_S000988*100


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
    soil_moisture_5cm_S000988 = model.predict(X_inputs_df, model='WeightedEnsemble_L2')  
    
    #return the numeric portion of the model prediction
    return soil_moisture_5cm_S000988[0] 
    #return d.atm_pressure_kPa*d.Soil_conductivity_5cm_S000988*100


@app.post('/predict')
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
    soil_moisture_5cm_S000988 = model.predict(X_inputs_df, model='WeightedEnsemble_L2')  
    
    #return the numeric portion of the model prediction
    return soil_moisture_5cm_S000988[0] 


@app.post('/batch/')
async def batch_inference(file: bytes = File(...)):
    # Correctly handle the uploaded file bytes
    data_df = pd.read_csv(BytesIO(file))
    
    # Optionally, you can preprocess the DataFrame if needed
    # For example, if you need to drop or modify any columns, do it here

    # Run inference using the model
    # Since the specifics of model.predict method depend on your model,
    # ensure to adapt this part to match your model's API
    

    predictions = model.predict(data_df, as_pandas=True)  # Assuming 'as_pandas' returns DataFrame
    
    # Convert predictions to JSON or any suitable format for API response
    return {
        "predictions": predictions.to_dict(orient='records')  # Converts DataFrame to a list of dicts
    }
    
    #return data_df.sum().sum()






    