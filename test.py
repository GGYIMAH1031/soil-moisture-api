
import requests, json

payload=json.dumps({
    "atm_pressure_kPa" : 98.81,
    "precipitation_mm" : 0,
    "Soil_conductivity_5cm_S000988" : 0.069,
    "radiation_W_m2": 223,
    "rel_humidity": 0.58,
    "Temp_2m_Celsius": 34.7,
    "windspeed_m_s": 1.18
})

#response=requests.put("http://127.0.0.1:8000/predict", data=payload)
response=requests.get("http://127.0.0.1:8000/predict", data=payload)
output = response.json()
print(output)