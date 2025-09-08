from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from typing import List, Dict, Any
import os
from sklearn.preprocessing import LabelEncoder

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
label_encoders = {}
scaler = None
feature_columns: list[str] = []

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess incoming data to match the training-time feature set and order."""
    global feature_columns, label_encoders

    df_processed = df.copy()

    # Drop known non-feature identifiers if present
    for col in ["id", "RowNumber", "Surname", "CustomerId"]:
        if col in df_processed.columns:
            df_processed = df_processed.drop(columns=[col])

    # Apply saved label encoders for any categorical/object columns
    for col, enc in (label_encoders or {}).items():
        if col in df_processed.columns:
            # Unseen labels -> set to first class to avoid errors
            try:
                df_processed[col] = enc.transform(df_processed[col].astype(str))
            except Exception:
                known = set(enc.classes_.tolist())
                df_processed[col] = df_processed[col].astype(str).apply(lambda v: v if v in known else enc.classes_[0])
                df_processed[col] = enc.transform(df_processed[col])

    # If we don't have encoders (cold start), fall back to simple maps for Geography/Gender
    if not label_encoders:
        if "Geography" in df_processed.columns:
            geography_map = {"France": 0, "Germany": 1, "Spain": 2}
            df_processed["Geography"] = df_processed["Geography"].map(geography_map).fillna(0)
        if "Gender" in df_processed.columns:
            gender_map = {"Female": 0, "Male": 1}
            df_processed["Gender"] = df_processed["Gender"].map(gender_map).fillna(0)

    # Align to training feature set: add missing columns with 0, drop extras, enforce order
    if feature_columns:
        for col in feature_columns:
            if col not in df_processed.columns:
                df_processed[col] = 0
        df_processed = df_processed[feature_columns]

    return df_processed

def create_dummy_model():
    """Create a simple dummy model for demonstration purposes"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    print("Creating dummy model for demonstration...")
    
    # Create dummy training data
    np.random.seed(42)
    n_samples = 1000
    
    # Create sample data that matches the expected format
    dummy_data = {
        'CreditScore': np.random.randint(300, 850, n_samples),
        'Geography': np.random.randint(0, 3, n_samples),
        'Gender': np.random.randint(0, 2, n_samples),
        'Age': np.random.randint(18, 95, n_samples),
        'Tenure': np.random.randint(0, 11, n_samples),
        'Balance': np.random.uniform(0, 250000, n_samples),
        'NumOfProducts': np.random.randint(1, 5, n_samples),
        'HasCrCard': np.random.randint(0, 2, n_samples),
        'IsActiveMember': np.random.randint(0, 2, n_samples),
        'EstimatedSalary': np.random.uniform(0, 200000, n_samples)
    }
    
    X_dummy = pd.DataFrame(dummy_data)
    # Create target with some logic (older customers with higher balance more likely to churn)
    y_dummy = ((X_dummy['Age'] > 50) & (X_dummy['Balance'] > 100000)).astype(int)
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_dummy, y_dummy)
    
    # Create scaler
    global scaler
    scaler = StandardScaler()
    scaler.fit(X_dummy)
    
    return model

def load_models():
    """Load trained models and their metadata"""
    global models, model_info, label_encoders, scaler, feature_columns
    
    model_dir = "../models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Try to load existing models
    model_files = {
        "xgboost": "xgboost_model.joblib",
        "lightgbm": "lightgbm_model.joblib", 
        "catboost": "catboost_model.joblib",
        "ensemble": "ensemble_model.joblib"
    }
    
    models_loaded = False
    
    for model_name, filename in model_files.items():
        model_path = os.path.join(model_dir, filename)
        if os.path.exists(model_path):
            try:
                models[model_name] = joblib.load(model_path)
                print(f"Loaded {model_name} model successfully")
                models_loaded = True
            except Exception as e:
                print(f"Error loading {model_name} model: {e}")
    
    # Try to load scaler and encoders
    try:
        scaler_path = os.path.join(model_dir, "scaler.joblib")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print("Loaded scaler successfully")
        
        encoders_path = os.path.join(model_dir, "label_encoders.joblib")
        if os.path.exists(encoders_path):
            label_encoders = joblib.load(encoders_path)
            print("Loaded label encoders successfully")
    except Exception as e:
        print(f"Error loading preprocessing objects: {e}")
    
    # If no models loaded, create a dummy model
    if not models_loaded:
        print("No trained models found. Creating dummy model for demonstration...")
        models["dummy"] = create_dummy_model()
        model_info = {
            "dummy": {
                "model_name": "dummy",
                "auc_score": 0.75,
                "accuracy": 0.80,
                "features": ["CreditScore", "Geography", "Gender", "Age", "Tenure", 
                           "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
            }
        }
    
    # Load model metadata if available
    metadata_path = os.path.join(model_dir, "model_metadata.json")
    if os.path.exists(metadata_path):
        import json
        with open(metadata_path, 'r') as f:
            meta_json = json.load(f)
            model_info = meta_json.get("model_performance", {})
            feature_columns = meta_json.get("feature_columns", feature_columns)

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

@app.get("/models", response_model=Dict[str, Any])
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
        
        # Preprocess the data
        df_processed = preprocess_data(df)
        
        # Scale if scaler is available
        if scaler is not None:
            df_scaled = scaler.transform(df_processed)
            df_processed = pd.DataFrame(df_scaled, columns=df_processed.columns)
        
        # Use the best available model (prefer ensemble, then others)
        model_priority = ["ensemble", "xgboost", "lightgbm", "catboost", "random_forest", "dummy"]
        model_name = None
        for name in model_priority:
            if name in models:
                model_name = name
                break
        
        if not model_name:
            raise HTTPException(status_code=500, detail="No suitable model found")
        
        model = models[model_name]
        
        # Make prediction
        prediction = model.predict(df_processed)[0]
        probability = model.predict_proba(df_processed)[0][1]  # Probability of churn
        
        # Determine confidence level
        if probability > 0.8 or probability < 0.2:
            confidence = "High"
        elif probability > 0.6 or probability < 0.4:
            confidence = "Medium"
        else:
            confidence = "Low"
        
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
        
        # Preprocess the data
        df_processed = preprocess_data(df)
        
        # Scale if scaler is available
        if scaler is not None:
            df_scaled = scaler.transform(df_processed)
            df_processed = pd.DataFrame(df_scaled, columns=df_processed.columns)
        
        # Use the best available model
        model_priority = ["ensemble", "xgboost", "lightgbm", "catboost", "random_forest", "dummy"]
        model_name = None
        for name in model_priority:
            if name in models:
                model_name = name
                break
        
        if not model_name:
            raise HTTPException(status_code=500, detail="No suitable model found")
        
        model = models[model_name]
        
        # Make predictions
        predictions = model.predict(df_processed)
        probabilities = model.predict_proba(df_processed)[:, 1]  # Probabilities of churn
        
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            if prob > 0.8 or prob < 0.2:
                confidence = "High"
            elif prob > 0.6 or prob < 0.4:
                confidence = "Medium"
            else:
                confidence = "Low"
            
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
        # Use the best available model
        model_priority = ["ensemble", "xgboost", "lightgbm", "catboost", "random_forest", "dummy"]
        model_name = None
        for name in model_priority:
            if name in models:
                model_name = name
                break
        
        if not model_name:
            raise HTTPException(status_code=500, detail="No suitable model found")
        
        model = models[model_name]
        
        # Get feature names
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
            # Create dummy importance for demonstration
            importance = np.random.random(len(feature_names))
            importance = importance / importance.sum()  # Normalize
        
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
