# Multi-stage Docker build for Customer Churn Prediction App

# Stage 1: Build React frontend
FROM node:18-alpine as frontend-builder

WORKDIR /app/frontend

# Copy package files
COPY frontend/package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY frontend/src ./src
COPY frontend/public ./public

# Build the React app
RUN npm run build

# Stage 2: Python backend
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ ./backend/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY models/ ./models/

# Copy built frontend from previous stage
COPY --from=frontend-builder /app/frontend/build ./frontend/build

# Create directories for data and models
RUN mkdir -p data models

# Expose ports
EXPOSE 8000 3000

# Set environment variables
ENV PYTHONPATH=/app
ENV REACT_APP_API_URL=http://localhost:8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command to run both backend and serve frontend
CMD ["sh", "-c", "cd backend && python main.py & cd ../frontend && npx serve -s build -l 3000 & wait"]
