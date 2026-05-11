from fastapi import FastAPI
import os
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

# Обучаем модель при старте
iris = load_iris()
model = RandomForestClassifier(n_estimators=100).fit(iris.data, iris.target)

@app.get("/health")
@app.get("/")
def health():
    # Берем версию из настроек Docker Compose
    return {"status": "ok", "version": os.getenv("APP_VERSION", "v1.0.0")}

@app.get("/predict")
def predict():
    # Пример предсказания для проверки
    pred = model.predict([iris.data[0]])
    return {"prediction": int(pred[0]), "status": "ok"}