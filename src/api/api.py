from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import configparser
import dagshub
import mlflow

# config
config_ini = configparser.ConfigParser()
config_ini.read('config.ini', encoding='utf-8')

# initialize dagshub
dagshub.auth.add_app_token(config_ini["dagshub"]["token"], host=None)
dagshub.init(repo_owner=config_ini["dagshub"]["user_name"], repo_name=config_ini["dagshub"]["project_name"], mlflow=True)

# initialize mlflow
client = mlflow.MlflowClient()
version = client.get_latest_versions(name=config_ini["dagshub"]["experiment_name"])[0].version
model_uri = f'models:/{config_ini["dagshub"]["experiment_name"]}/{version}'

# load model
model = mlflow.sklearn.load_model(model_uri)

# FastAPI アプリケーションの作成
app = FastAPI()

# 入力データのバリデーション
class InputData(BaseModel):
    features: list[float]

@app.post("/predict/")
def predict(data: InputData):
    X_input = np.array(data.features).reshape(1, -1)
    if X_input.shape[1] != 2:
        raise HTTPException(status_code=400, detail="Input must have exactly 2 features")
    
    prediction = model.predict(X_input)
    return {"prediction": int(prediction[0])}
