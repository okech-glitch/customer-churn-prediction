#!/usr/bin/env python3
"""
Customer Churn Prediction - Setup Script
Automates the setup and training process for the churn prediction system
"""

import os
import sys
import subprocess
import json
import pandas as pd
from pathlib import Path

def run_command(command, cwd=None):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            cwd=cwd, 
            capture_output=True, 
            text=True, 
            check=True
        )
        print(f"✓ {command}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"✗ {command}")
        print(f"Error: {e.stderr}")
        return None

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✓ Python {sys.version.split()[0]} detected")

def check_node_version():
    """Check if Node.js is installed"""
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Node.js {result.stdout.strip()} detected")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ Node.js not found. Please install Node.js 16 or higher")
    print("Download from: https://nodejs.org/")
    return False

def setup_python_environment():
    """Set up Python environment and install dependencies"""
    print("\n🐍 Setting up Python environment...")
    
    # Install Python dependencies
    if run_command("pip install -r requirements.txt"):
        print("✓ Python dependencies installed")
    else:
        print("❌ Failed to install Python dependencies")
        return False
    
    return True

def setup_frontend():
    """Set up React frontend"""
    print("\n⚛️ Setting up React frontend...")
    
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return False
    
    # Install Node.js dependencies
    if run_command("npm install", cwd=frontend_dir):
        print("✓ Frontend dependencies installed")
    else:
        print("❌ Failed to install frontend dependencies")
        return False
    
    return True

def create_sample_data():
    """Create sample data if not present"""
    print("\n📊 Creating sample data...")
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Check if data files exist
    train_file = data_dir / "train.csv"
    test_file = data_dir / "test.csv"
    
    if not train_file.exists() or not test_file.exists():
        print("⚠️  Data files not found. Please download the competition data:")
        print("   1. Go to: https://www.kaggle.com/competitions/binaryclassificationwithabankchurndatasetumgc")
        print("   2. Download train.csv and test.csv")
        print("   3. Place them in the data/ directory")
        return False
    
    print("✓ Data files found")
    return True

def train_models():
    """Train the machine learning models"""
    print("\n🤖 Training machine learning models...")
    
    if run_command("python scripts/train_models.py"):
        print("✓ Models trained successfully")
        return True
    else:
        print("❌ Model training failed")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating project directories...")
    
    directories = [
        "data",
        "models", 
        "notebooks",
        "scripts",
        "tests",
        "backend",
        "frontend/src",
        "frontend/public"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {directory}/")

def create_env_file():
    """Create environment configuration file"""
    print("\n⚙️ Creating environment configuration...")
    
    env_content = """# Customer Churn Prediction Environment Configuration

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_URL=http://localhost:8000

# Frontend Configuration
REACT_APP_API_URL=http://localhost:8000
REACT_APP_ENVIRONMENT=development

# Model Configuration
MODEL_PATH=./models
DATA_PATH=./data

# Database Configuration (Optional)
DATABASE_URL=sqlite:///./churn_prediction.db
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("✓ Environment file created")

def print_usage_instructions():
    """Print instructions for using the application"""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    print("\n📋 Next Steps:")
    print("\n1. Start the Backend API:")
    print("   cd backend")
    print("   python main.py")
    print("\n2. Start the Frontend (in a new terminal):")
    print("   cd frontend")
    print("   npm start")
    print("\n3. Open your browser and go to:")
    print("   http://localhost:3000")
    print("\n🐳 Alternative - Use Docker:")
    print("   docker-compose up --build")
    print("\n📊 Run Data Analysis:")
    print("   jupyter notebook notebooks/data_analysis.ipynb")
    print("\n🔧 Train New Models:")
    print("   python scripts/train_models.py")
    print("\n" + "="*60)

def main():
    """Main setup function"""
    print("🚀 Customer Churn Prediction - Setup Script")
    print("="*50)
    
    # Check system requirements
    check_python_version()
    has_node = check_node_version()
    
    # Create project structure
    create_directories()
    create_env_file()
    
    # Set up Python environment
    if not setup_python_environment():
        print("❌ Python setup failed")
        sys.exit(1)
    
    # Set up frontend (if Node.js is available)
    if has_node:
        if not setup_frontend():
            print("⚠️  Frontend setup failed, but continuing...")
    else:
        print("⚠️  Skipping frontend setup (Node.js not available)")
    
    # Check for data files
    if not create_sample_data():
        print("⚠️  Please add data files before training models")
        print_usage_instructions()
        return
    
    # Train models
    if not train_models():
        print("⚠️  Model training failed, but setup is complete")
        print("   You can run 'python scripts/train_models.py' later")
    
    # Print usage instructions
    print_usage_instructions()

if __name__ == "__main__":
    main()
