# Customer Churn Prediction - Fullstack ML Project

A comprehensive fullstack machine learning project for predicting bank customer churn using multiple ML algorithms and a modern web interface.

## 🎯 Project Overview

This project predicts whether bank customers will churn (leave) or stay based on various customer features. It includes:

- **Data Analysis & Preprocessing**: Comprehensive EDA and feature engineering
- **Multiple ML Models**: XGBoost, LightGBM, CatBoost, and ensemble methods
- **FastAPI Backend**: RESTful API for model serving
- **React Frontend**: Interactive dashboard for predictions and analysis
- **Model Evaluation**: ROC-AUC scoring and cross-validation

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies (for frontend)
cd frontend
npm install
```

### 2. Prepare Data
Place your Kaggle competition files in the `data/` directory:
- `train.csv`
- `test.csv` 
- `sample_submission.csv`

### 3. Run the Project

**Backend (API Server):**
```bash
cd backend
python main.py
```

**Frontend (React Dashboard):**
```bash
cd frontend
npm start
```

**Jupyter Notebooks (Analysis):**
```bash
jupyter notebook notebooks/
```

## 📁 Project Structure

```
customer-churn-prediction/
├── backend/                 # FastAPI backend
│   ├── main.py             # API server
│   ├── models/             # ML model files
│   └── utils/              # Utility functions
├── frontend/               # React frontend
│   ├── src/                # React components
│   ├── public/             # Static assets
│   └── package.json        # Node dependencies
├── data/                   # Dataset files
├── models/                 # Trained model artifacts
├── notebooks/              # Jupyter notebooks for analysis
├── scripts/                # Data processing scripts
└── tests/                  # Unit tests
```

## 🧠 Machine Learning Pipeline

1. **Data Preprocessing**: Handle missing values, encode categorical variables
2. **Feature Engineering**: Create new features, scale numerical variables
3. **Model Training**: Train multiple algorithms (XGBoost, LightGBM, CatBoost)
4. **Ensemble Methods**: Combine models for better performance
5. **Model Evaluation**: Cross-validation and ROC-AUC scoring
6. **Prediction**: Generate predictions for test set

## 📊 Features

- **Interactive Dashboard**: Visualize predictions and model performance
- **Real-time Predictions**: API endpoints for single and batch predictions
- **Model Comparison**: Compare different algorithms side-by-side
- **Feature Importance**: Understand which features drive churn predictions
- **Export Results**: Generate submission files for Kaggle

## 🎯 Competition Details

- **Evaluation Metric**: Area under the ROC curve (AUC)
- **Target Variable**: Binary classification (0 = Stay, 1 = Churn)
- **Submission Format**: CSV with id and Exited probability columns

## 🔧 Technologies Used

- **Backend**: FastAPI, SQLAlchemy, Scikit-learn
- **Frontend**: React, TypeScript, Material-UI
- **ML**: XGBoost, LightGBM, CatBoost, Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Deployment**: Docker, SQLite

## 📈 Performance Goals

- Target AUC Score: > 0.85
- Model Interpretability: Feature importance analysis
- Scalability: Handle batch predictions efficiently
- User Experience: Intuitive dashboard interface

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.
