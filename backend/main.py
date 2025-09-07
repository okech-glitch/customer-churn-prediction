from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from typing import List, Dict, Any
import os

app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class CustomerData(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

class PredictionResponse(BaseModel):
    customer_id: int
    churn_probability: float
    prediction: int
    confidence: str

class BatchPredictionRequest(BaseModel):
    customers: List[CustomerData]

class ModelInfo(BaseModel):
    model_name: str
    accuracy: float
    auc_score: float
    features: List[str]

# Global variables for models
models = {}
model_info = {}

def load_models():
    """Load trained models and their metadata"""
    global models, model_info
    
    model_dir = "../models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Try to load different model types
    model_files = {
        "xgboost": "xgboost_model.joblib",
        "lightgbm": "lightgbm_model.joblib", 
        "catboost": "catboost_model.joblib",
        "ensemble": "ensemble_model.joblib"
    }
    
    for model_name, filename in model_files.items():
        model_path = os.path.join(model_dir, filename)
        if os.path.exists(model_path):
            try:
                models[model_name] = joblib.load(model_path)
                print(f"Loaded {model_name} model successfully")
            except Exception as e:
                print(f"Error loading {model_name} model: {e}")
    
    # Load model metadata
    metadata_path = os.path.join(model_dir, "model_metadata.json")
    if os.path.exists(metadata_path):
        import json
        with open(metadata_path, 'r') as f:
            model_info = json.load(f)

# Load models on startup
@app.on_event("startup")
async def startup_event():
    load_models()

@app.get("/")
async def root():
    return {"message": "Customer Churn Prediction API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": len(models)}

@app.get("/models", response_model=Dict[str, ModelInfo])
async def get_models():
    """Get information about loaded models"""
    if not models:
        raise HTTPException(status_code=404, detail="No models loaded")
    
    return model_info

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(customer: CustomerData):
    """Predict churn for a single customer"""
    if not models:
        raise HTTPException(status_code=500, detail="No models loaded")
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame([customer.dict()])
        
        # Use the best available model (prefer ensemble)
        model_name = "ensemble" if "ensemble" in models else list(models.keys())[0]
        model = models[model_name]
        
        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]  # Probability of churn
        
        # Determine confidence level
        confidence = "High" if probability > 0.8 or probability < 0.2 else "Medium" if probability > 0.6 or probability < 0.4 else "Low"
        
        return PredictionResponse(
            customer_id=0,  # Will be assigned by frontend
            churn_probability=float(probability),
            prediction=int(prediction),
            confidence=confidence
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """Predict churn for multiple customers"""
    if not models:
        raise HTTPException(status_code=500, detail="No models loaded")
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame([customer.dict() for customer in request.customers])
        
        # Use the best available model
        model_name = "ensemble" if "ensemble" in models else list(models.keys())[0]
        model = models[model_name]
        
        # Make predictions
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]  # Probabilities of churn
        
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            confidence = "High" if prob > 0.8 or prob < 0.2 else "Medium" if prob > 0.6 or prob < 0.4 else "Low"
            
            results.append({
                "customer_id": i,
                "churn_probability": float(prob),
                "prediction": int(pred),
                "confidence": confidence
            })
        
        return {"predictions": results, "model_used": model_name}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

@app.get("/features")
async def get_feature_importance():
    """Get feature importance from the best model"""
    if not models:
        raise HTTPException(status_code=404, detail="No models loaded")
    
    try:
        model_name = "ensemble" if "ensemble" in models else list(models.keys())[0]
        model = models[model_name]
        
        # Get feature names (assuming they match the training data)
        feature_names = [
            "CreditScore", "Geography", "Gender", "Age", "Tenure", 
            "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
        ]
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = abs(model.coef_[0])
        else:
            raise HTTPException(status_code=500, detail="Model doesn't support feature importance")
        
        # Create feature importance dictionary
        feature_importance = dict(zip(feature_names, importance.tolist()))
        
        return {
            "model_name": model_name,
            "feature_importance": feature_importance
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting feature importance: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
