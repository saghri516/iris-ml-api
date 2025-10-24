from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load trained Decision Tree model
model = joblib.load("model.joblib")

app = FastAPI(title="Decision Tree ML Model API", version="1.0")

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

labels = ["setosa", "versicolor", "virginica"]

@app.get("/")
def home():
    return {"message": "Decision Tree Iris Model is running!"}

@app.post("/predict")
def predict(data: IrisFeatures):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]
    prediction = model.predict(features)[0]
    return {"prediction": labels[prediction]}

@app.get("/health")
def health():
    return {"status":"ok"}
