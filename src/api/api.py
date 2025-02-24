from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

# モデルのトレーニングと保存
def train_and_save_model():
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    y = np.array([0, 1, 1, 1])
    model = LogisticRegression()
    model.fit(X, y)
    joblib.dump(model, "model.pkl")

# モデルの読み込み
try:
    model = joblib.load("model.pkl")
except FileNotFoundError:
    train_and_save_model()
    model = joblib.load("model.pkl")

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
