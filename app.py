from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
from src.mlProject.pipeline.prediction import PredictionPipeline
from src.mlProject import logger
import subprocess

app = FastAPI()

# Prediction Instance
prediction_pipeline = PredictionPipeline()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return """
    <html>
        <head><title>Telco Churn API</title></head>
        <body>
            <h1>Telco Customer Churn Prediction API</h1>
            <p>Endpoints:</p>
            <ul>
                <li>POST /predict : Predict churn for a customer</li>
                <li>GET /train : Trigger model training pipeline</li>
                <li>GET /health : Health check</li>
            </ul>
        </body>
    </html>
    """

@app.get("/train")
async def train():
    try:
        os.system("python main.py")
        return JSONResponse(content={"message": "Training successful!"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

from pydantic import BaseModel

class CustomerData(BaseModel):
    gender: str = "Female"
    SeniorCitizen: int = 0
    Partner: str = "Yes"
    Dependents: str = "No"
    tenure: int = 1
    PhoneService: str = "No"
    MultipleLines: str = "No phone service"
    InternetService: str = "DSL"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "Yes"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 29.85
    TotalCharges: float = 29.85

@app.post("/predict")
async def predict_route(data: CustomerData):
    try:
        # Convert Pydantic model to Dict
        input_dict = data.dict()
        prediction, probability = prediction_pipeline.predict(input_dict)
        
        result = {
            "prediction": int(prediction),
            "churn": "Yes" if prediction == 1 else "No",
            "probability": float(probability)
        }
        return JSONResponse(content=result)
    except Exception as e:
        logger.exception(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
