"""
Simplified Customer Churn Training Script
Uses only scikit-learn to avoid dependency issues
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import json
import os
from datetime import datetime

def main():
    print("🚀 Starting simplified model training...")
    
    # Load data
    train_data = pd.read_csv('../data/train.csv')
    test_data = pd.read_csv('../data/test.csv')
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Preprocess
    # Drop obvious non-feature identifiers if present
    for col_to_drop in ['id', 'Surname']:
        if col_to_drop in train_data.columns:
            # keep id in test separately for submission
            if col_to_drop == 'id':
                pass
            else:
                train_data = train_data.drop(columns=[col_to_drop])
        if col_to_drop in test_data.columns and col_to_drop != 'id':
            test_data = test_data.drop(columns=[col_to_drop])

    # Label-encode all object dtype columns using combined fit (train+test)
    label_encoders = {}
    object_cols = list(set(train_data.select_dtypes(include=['object']).columns)
                       .union(set(test_data.select_dtypes(include=['object']).columns)) - set(['Exited']))

    for feature in object_cols:
        combined = pd.concat([
            train_data[feature].astype(str),
            test_data[feature].astype(str)
        ], axis=0)
        le = LabelEncoder()
        le.fit(combined)
        train_data[feature] = le.transform(train_data[feature].astype(str))
        test_data[feature] = le.transform(test_data[feature].astype(str))
        label_encoders[feature] = le
    
    # Features and target
    feature_columns = [col for col in train_data.columns if col not in ['id', 'Exited']]
    X = train_data[feature_columns]
    y = train_data['Exited']
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train models
    models = {
        'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_val_scaled)
        y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
        auc_score = roc_auc_score(y_val, y_pred_proba)
        
        results[name] = {
            'auc_score': auc_score,
            'model': model
        }
        
        print(f"{name} AUC: {auc_score:.4f}")
    
    # Create ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('rf', results['random_forest']['model']),
            ('lr', results['logistic_regression']['model'])
        ],
        voting='soft'
    )
    ensemble.fit(X_train_scaled, y_train)
    
    y_pred_ensemble = ensemble.predict(X_val_scaled)
    y_pred_proba_ensemble = ensemble.predict_proba(X_val_scaled)[:, 1]
    auc_ensemble = roc_auc_score(y_val, y_pred_proba_ensemble)
    
    print(f"Ensemble AUC: {auc_ensemble:.4f}")
    
    # Save models
    os.makedirs('../models', exist_ok=True)
    
    for name, result in results.items():
        joblib.dump(result['model'], f'../models/{name}_model.joblib')
    
    joblib.dump(ensemble, '../models/ensemble_model.joblib')
    joblib.dump(scaler, '../models/scaler.joblib')
    joblib.dump(label_encoders, '../models/label_encoders.joblib')
    
    # Save metadata
    metadata = {
        'feature_columns': feature_columns,
        'model_performance': {
            name: {'auc_score': result['auc_score']} 
            for name, result in results.items()
        },
        'ensemble_auc': auc_ensemble,
        'training_date': datetime.now().isoformat()
    }
    
    with open('../models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Generate predictions
    X_test = test_data[feature_columns]
    X_test_scaled = scaler.transform(X_test)
    
    predictions = ensemble.predict_proba(X_test_scaled)[:, 1]
    
    submission = pd.DataFrame({
        'id': test_data['id'],
        'Exited': predictions
    })
    
    submission.to_csv('../data/submission.csv', index=False)
    
    print("✅ Training complete!")
    print(f"Best model: ensemble (AUC: {auc_ensemble:.4f})")
    print("Models saved to ../models/")
    print("Predictions saved to ../data/submission.csv")

if __name__ == "__main__":
    main()
