"""
Customer Churn Prediction - Model Training Script
Trains multiple ML models and creates ensemble for churn prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

class ChurnPredictor:
    def __init__(self, data_path="../data"):
        self.data_path = data_path
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        
    def load_data(self):
        """Load and inspect the dataset"""
        print("Loading data...")
        
        # Load training data
        train_path = os.path.join(self.data_path, "train.csv")
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data not found at {train_path}")
        
        self.train_data = pd.read_csv(train_path)
        print(f"Training data shape: {self.train_data.shape}")
        
        # Load test data
        test_path = os.path.join(self.data_path, "test.csv")
        if os.path.exists(test_path):
            self.test_data = pd.read_csv(test_path)
            print(f"Test data shape: {self.test_data.shape}")
        
        # Display basic info
        print("\nDataset Info:")
        print(self.train_data.info())
        print("\nFirst few rows:")
        print(self.train_data.head())
        
        return self.train_data, self.test_data if hasattr(self, 'test_data') else None
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Target variable distribution
        print("\nTarget Variable Distribution:")
        print(self.train_data['Exited'].value_counts())
        print(f"Churn Rate: {self.train_data['Exited'].mean():.2%}")
        
        # Numerical features analysis
        numerical_features = self.train_data.select_dtypes(include=[np.number]).columns
        print(f"\nNumerical Features: {list(numerical_features)}")
        
        # Categorical features analysis
        categorical_features = self.train_data.select_dtypes(include=['object']).columns
        print(f"Categorical Features: {list(categorical_features)}")
        
        # Missing values
        print("\nMissing Values:")
        missing_values = self.train_data.isnull().sum()
        print(missing_values[missing_values > 0])
        
        # Correlation analysis
        plt.figure(figsize=(12, 8))
        correlation_matrix = self.train_data[numerical_features].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('../notebooks/correlation_matrix.png')
        plt.show()
        
        return numerical_features, categorical_features
    
    def preprocess_data(self):
        """Preprocess the data for training"""
        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)
        
        # Create a copy for preprocessing
        train_processed = self.train_data.copy()
        test_processed = self.test_data.copy() if hasattr(self, 'test_data') else None
        
        # Handle categorical variables
        categorical_features = ['Geography', 'Gender']
        self.label_encoders = {}
        
        for feature in categorical_features:
            le = LabelEncoder()
            train_processed[feature] = le.fit_transform(train_processed[feature])
            if test_processed is not None:
                test_processed[feature] = le.transform(test_processed[feature])
            self.label_encoders[feature] = le
        
        # Separate features and target
        feature_columns = [col for col in train_processed.columns if col not in ['id', 'Exited']]
        self.feature_columns = feature_columns
        
        X = train_processed[feature_columns]
        y = train_processed['Exited']
        
        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale numerical features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Convert back to DataFrames
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_columns, index=X_train.index)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_columns, index=X_val.index)
        
        self.X_train, self.X_val = X_train_scaled, X_val_scaled
        self.y_train, self.y_val = y_train, y_val
        
        print(f"Training set shape: {X_train_scaled.shape}")
        print(f"Validation set shape: {X_val_scaled.shape}")
        print(f"Features: {feature_columns}")
        
        return X_train_scaled, X_val_scaled, y_train, y_val
    
    def train_models(self):
        """Train multiple ML models"""
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)
        
        # Define models
        models_config = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            ),
            'catboost': CatBoostClassifier(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=False
            )
        }
        
        # Train each model
        for name, model in models_config.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_val)
            y_pred_proba = model.predict_proba(self.X_val)[:, 1]
            
            # Calculate metrics
            auc_score = roc_auc_score(self.y_val, y_pred_proba)
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='roc_auc')
            
            # Store results
            self.models[name] = model
            self.results[name] = {
                'auc_score': auc_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"AUC Score: {auc_score:.4f}")
            print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Create ensemble model
        print("\nCreating ensemble model...")
        ensemble_models = [
            ('rf', self.models['random_forest']),
            ('xgb', self.models['xgboost']),
            ('lgb', self.models['lightgbm']),
            ('cat', self.models['catboost'])
        ]
        
        self.models['ensemble'] = VotingClassifier(
            estimators=ensemble_models,
            voting='soft'
        )
        
        self.models['ensemble'].fit(self.X_train, self.y_train)
        
        # Evaluate ensemble
        y_pred_ensemble = self.models['ensemble'].predict(self.X_val)
        y_pred_proba_ensemble = self.models['ensemble'].predict_proba(self.X_val)[:, 1]
        auc_ensemble = roc_auc_score(self.y_val, y_pred_proba_ensemble)
        
        self.results['ensemble'] = {
            'auc_score': auc_ensemble,
            'predictions': y_pred_ensemble,
            'probabilities': y_pred_proba_ensemble
        }
        
        print(f"Ensemble AUC Score: {auc_ensemble:.4f}")
        
        return self.models, self.results
    
    def evaluate_models(self):
        """Evaluate and compare all models"""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            name: {
                'AUC Score': result['auc_score'],
                'CV Mean': result.get('cv_mean', 'N/A'),
                'CV Std': result.get('cv_std', 'N/A')
            }
            for name, result in self.results.items()
        }).T
        
        print("\nModel Performance Comparison:")
        print(results_df.round(4))
        
        # Find best model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['auc_score'])
        print(f"\nBest Model: {best_model_name} (AUC: {self.results[best_model_name]['auc_score']:.4f})")
        
        # Plot ROC curves
        plt.figure(figsize=(10, 8))
        for name, result in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_val, result['probabilities'])
            plt.plot(fpr, tpr, label=f'{name} (AUC = {result["auc_score"]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('../notebooks/roc_curves.png')
        plt.show()
        
        return results_df, best_model_name
    
    def get_feature_importance(self):
        """Extract feature importance from models"""
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE")
        print("="*50)
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = abs(model.coef_[0])
            else:
                continue
            
            self.feature_importance[name] = dict(zip(self.feature_columns, importance))
        
        # Plot feature importance for best tree-based models
        tree_models = ['random_forest', 'xgboost', 'lightgbm', 'catboost']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, model_name in enumerate(tree_models):
            if model_name in self.feature_importance:
                importance_df = pd.DataFrame(
                    list(self.feature_importance[model_name].items()),
                    columns=['Feature', 'Importance']
                ).sort_values('Importance', ascending=True)
                
                axes[i].barh(importance_df['Feature'], importance_df['Importance'])
                axes[i].set_title(f'{model_name.title()} Feature Importance')
                axes[i].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig('../notebooks/feature_importance.png')
        plt.show()
        
        return self.feature_importance
    
    def save_models(self):
        """Save trained models and metadata"""
        print("\n" + "="*50)
        print("SAVING MODELS")
        print("="*50)
        
        # Create models directory
        models_dir = "../models"
        os.makedirs(models_dir, exist_ok=True)
        
        # Save individual models
        for name, model in self.models.items():
            model_path = os.path.join(models_dir, f"{name}_model.joblib")
            joblib.dump(model, model_path)
            print(f"Saved {name} model to {model_path}")
        
        # Save scaler and encoders
        joblib.dump(self.scaler, os.path.join(models_dir, "scaler.joblib"))
        joblib.dump(self.label_encoders, os.path.join(models_dir, "label_encoders.joblib"))
        
        # Save metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'model_performance': {
                name: {
                    'auc_score': result['auc_score'],
                    'cv_mean': result.get('cv_mean', None),
                    'cv_std': result.get('cv_std', None)
                }
                for name, result in self.results.items()
            },
            'training_date': datetime.now().isoformat(),
            'data_shape': self.train_data.shape
        }
        
        with open(os.path.join(models_dir, "model_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved model metadata to {os.path.join(models_dir, 'model_metadata.json')}")
    
    def generate_predictions(self):
        """Generate predictions for test set"""
        if not hasattr(self, 'test_data'):
            print("No test data available for predictions")
            return None
        
        print("\n" + "="*50)
        print("GENERATING PREDICTIONS")
        print("="*50)
        
        # Preprocess test data
        test_processed = self.test_data.copy()
        
        # Apply label encoding
        for feature, encoder in self.label_encoders.items():
            test_processed[feature] = encoder.transform(test_processed[feature])
        
        # Select features and scale
        X_test = test_processed[self.feature_columns]
        X_test_scaled = self.scaler.transform(X_test)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_columns)
        
        # Use best model for predictions
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['auc_score'])
        best_model = self.models[best_model_name]
        
        # Generate predictions
        predictions = best_model.predict_proba(X_test_scaled)[:, 1]
        
        # Create submission file
        submission = pd.DataFrame({
            'id': self.test_data['id'],
            'Exited': predictions
        })
        
        submission_path = "../data/submission.csv"
        submission.to_csv(submission_path, index=False)
        print(f"Saved predictions to {submission_path}")
        
        return submission

def main():
    """Main training pipeline"""
    print("Customer Churn Prediction - Model Training")
    print("="*50)
    
    # Initialize predictor
    predictor = ChurnPredictor()
    
    # Load data
    train_data, test_data = predictor.load_data()
    
    # Explore data
    predictor.explore_data()
    
    # Preprocess data
    predictor.preprocess_data()
    
    # Train models
    predictor.train_models()
    
    # Evaluate models
    results_df, best_model = predictor.evaluate_models()
    
    # Get feature importance
    predictor.get_feature_importance()
    
    # Save models
    predictor.save_models()
    
    # Generate predictions
    submission = predictor.generate_predictions()
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print(f"Best model: {best_model}")
    print(f"Best AUC score: {predictor.results[best_model]['auc_score']:.4f}")
    print("Models saved to ../models/")
    print("Predictions saved to ../data/submission.csv")

if __name__ == "__main__":
    main()
