# 🍎 Fruit Classification ML Pipeline

End-to-end machine learning pipeline for real-time fruit image classification with model monitoring, prediction API, and automatic retraining capabilities.

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Features](#-features)
3. [Project Structure](#-project-structure)
4. [Installation](#-installation)
5. [Quick Start](#-quick-start)
6. [API Endpoints](#-api-endpoints)
7. [Usage Examples](#-usage-examples)
8. [Load Testing Results](#-load-testing-results)
9. [Docker Deployment](#-docker-deployment)
10. [Video Demo](#-video-demo)
11. [Model Evaluation](#-model-evaluation)

---

## 🎯 Project Overview

This project implements a complete ML pipeline for fruit classification using:

- **Model**: Transfer learning with MobileNetV2
- **Framework**: TensorFlow/Keras
- **API**: Flask REST API with CORS
- **Frontend**: Interactive web dashboard
- **Deployment**: Docker containerization
- **Monitoring**: Real-time metrics tracking
- **Retraining**: Automated model updating with new data

**Supported Fruits**: Apple, Banana, Cherry, Coconut, Grape, Guava, Kiwi, Mango, Orange, Papaya, Pineapple, Strawberry

---

## ✨ Features

### 🔮 Prediction
- ✅ Single image classification
- ✅ Batch predictions
- ✅ Confidence scores and probabilities
- ✅ Real-time inference (<100ms response time)

### 📊 Monitoring
- ✅ Model uptime tracking
- ✅ Request counting
- ✅ Response time metrics
- ✅ Throughput measurement
- ✅ Health checks

### 🔄 Retraining
- ✅ Bulk data upload (ZIP files)
- ✅ Automatic model updating
- ✅ Background processing
- ✅ Model versioning with backups

### 🎨 Web UI
- ✅ Interactive dashboard
- ✅ Real-time metrics display
- ✅ Image preview
- ✅ Batch processing interface
- ✅ Responsive design

### 🐳 Deployment
- ✅ Docker containerization
- ✅ Multi-container orchestration
- ✅ Nginx reverse proxy
- ✅ Volume mounting for persistence

### 📈 Testing
- ✅ Locust load testing
- ✅ Performance metrics
- ✅ Concurrent user simulation
- ✅ Response time analysis

---

## 📁 Project Structure

```
fruit-ml-pipeline/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── Dockerfile                          # Container configuration
├── docker-compose.yml                  # Multi-container setup
├── locustfile.py                       # Load testing script
│
├── notebook/
│   └── fruit_classification.ipynb      # Jupyter notebook with full pipeline
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                # Data loading & preprocessing
│   ├── model.py                        # Model building & training
│   ├── prediction.py                   # Inference module
│   ├── retraining.py                   # Model retraining pipeline
│   └── api.py                          # Flask REST API
│
├── frontend/
│   ├── index.html                      # Web dashboard
│   ├── style.css                       # Styling
│   └── script.js                       # Interactive functionality
│
├── data/
│   ├── train/                          # Training images by class
│   │   ├── apple/
│   │   ├── banana/
│   │   └── ...
│   └── test/                           # Test images
│
├── models/
│   ├── fruit_classifier_model.h5       # Trained model
│   ├── fruit_classes.pkl               # Class names
│   └── metrics.pkl                     # Model metrics
│
├── uploads/                            # Temporary uploaded files
├── visualizations/                     # Generated visualizations
└── logs/                               # API logs
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10+
- Docker & Docker Compose (optional)
- 4GB RAM minimum
- 2GB disk space

### Step 1: Clone Repository

```bash
git https://github.com/belladev250/Fruit-predicition-pipeline.git
cd Fruit-predicition-pipeline.git
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset

```bash
# Using Kaggle CLI
kaggle datasets download -d moltean/fruits
unzip fruits.zip -d data/

# OR manual download:
# Visit: https://www.kaggle.com/datasets/moltean/fruits
```

### Step 5: Verify Dataset Structure

```bash
# Should have this structure:
data/Fruit-Images-Dataset-Original-Size/
├── apple/
├── banana/
├── cherry/
├── coconut/
├── grape/
├── guava/
├── kiwi/
├── mango/
├── orange/
├── papaya/
├── pineapple/
└── strawberry/
```

---

## ⚡ Quick Start

### Option 1: Run Everything (Jupyter Notebook)

```bash
jupyter notebook notebook/fruit_classification.ipynb
```

This runs the complete pipeline:
1. Data loading & preprocessing
2. 3 feature visualizations with interpretations
3. Model training with optimization techniques
4. Model evaluation (4+ metrics)
5. Model saving

### Option 2: Run API Only

```bash
python src/api.py
```

Server starts at: `http://localhost:5000`

### Option 3: Run with Docker

```bash
# Build image
docker build -t fruit-classifier .

# Run single container
docker run -p 5000:5000 fruit-classifier

# OR run with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f api
```

### Option 4: Access Web Dashboard

Once API is running, open browser:
```
http://localhost  # If using Docker/Nginx
http://localhost:5000  # If running API directly
```

---

## 📡 API Endpoints

### Health & Status

```bash
GET /health
GET /status
GET /info
```

### Predictions

```bash
POST /predict
# Upload single image and get prediction

POST /predict-batch
# Upload multiple images and get predictions

GET /model-info
# Get model classes and configuration
```

### Metrics

```bash
GET /metrics
# Get API performance metrics

GET /model-info
# Get model details
```

### Retraining

```bash
POST /retrain
# Trigger model retraining with uploaded ZIP

GET /retrain-status
# Check retraining progress
```

### Data Upload

```bash
POST /upload-data
# Upload training data for retraining
```

---

## 💡 Usage Examples

### Example 1: Single Prediction

```bash
curl -X POST -F "image=@fruit.jpg" http://localhost:5000/predict
```

**Response:**
```json
{
  "prediction": {
    "class": "apple",
    "confidence": 0.95,
    "probabilities": {
      "apple": 0.95,
      "banana": 0.02,
      "cherry": 0.03
    }
  },
  "response_time_ms": 45.23
}
```

### Example 2: Batch Prediction

```bash
curl -X POST \
  -F "images=@apple.jpg" \
  -F "images=@banana.jpg" \
  -F "images=@cherry.jpg" \
  http://localhost:5000/predict-batch
```

### Example 3: Trigger Retraining

```bash


curl -X POST -F "file=@new_data.zip" http://localhost:5000/retrain
```

### Example 4: Get Metrics

```bash
curl http://localhost:5000/metrics
```

**Response:**
```json
{
  "uptime_seconds": 3600.5,
  "total_requests": 1523,
  "average_response_time_ms": 67.34,
  "requests_per_minute": 25.38,
  "retraining": false
}
```

---

## 📊 Model Evaluation

### Training Results

```
🎯 Accuracy:  0.9969 (99.69%)
📏 Precision: 1.0000
📈 Recall:    0.9969
⭐ F1 Score:  0.9984

### Evaluation Metrics Used

1. **Accuracy**: Overall correct predictions
2. **Precision**: True positives / (True positives + False positives)
3. **Recall**: True positives / (True positives + False negatives)
4. **F1 Score**: Harmonic mean of Precision and Recall
5. **Confusion Matrix**: Per-class performance
6. **Classification Report**: Detailed per-class metrics

### Model Architecture

- **Base Model**: MobileNetV2 (ImageNet pretrained)
- **Custom Layers**:
  - Global Average Pooling
  - Dense 256 (ReLU) + BatchNorm + Dropout(0.5)
  - Dense 128 (ReLU) + BatchNorm + Dropout(0.3)
  - Dense 12 (Softmax)
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Sparse Categorical Crossentropy

### Optimization Techniques

- ✅ Transfer Learning (MobileNetV2)
- ✅ Batch Normalization
- ✅ Dropout Regularization (0.5, 0.3)
- ✅ Early Stopping
- ✅ Learning Rate Reduction

---

## 🐳 Docker Deployment

### Single Container

```bash
docker build -t fruit-classifier .
docker run -p 5000:5000 -p 80:80 fruit-classifier
```

### Multi-Container with Docker Compose

```bash
# Start services
docker-compose up -d

# Scale API to 3 containers
docker-compose up -d --scale api=3

# View status
docker-compose ps

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

---

## 🔥 Load Testing Results

### Test Configuration

```bash
# Command:
locust -f locustfile.py --host=http://localhost:5000 -u 100 -r 10 -t 5m

# Parameters:
# -u 100 = 100 concurrent users
# -r 10 = Spawn 10 users per second  
# -t 5m = Run for 5 minutes
```

### Results - Single Container

```
Total Requests: 5,432
Requests/min: 1,086
Average Response Time: 127ms
95th Percentile: 245ms
99th Percentile: 523ms
Min: 45ms
Max: 1234ms
```

### Results - 2 Containers

```
Total Requests: 7,821
Requests/min: 1,564
Average Response Time: 89ms
95th Percentile: 167ms
99th Percentile: 298ms
Min: 34ms
Max: 734ms
```

### Results - 3 Containers

```
Total Requests: 9,143
Requests/min: 1,829
Average Response Time: 72ms
95th Percentile: 134ms
99th Percentile: 201ms
Min: 28ms
Max: 521ms
```

### Running Load Tests

```bash
# Light load
locust -f locustfile.py --host=http://localhost:5000 -u 10 -r 5 -t 5m

# Medium load
locust -f locustfile.py --host=http://localhost:5000 -u 50 -r 10 -t 5m

# Heavy load
locust -f locustfile.py --host=http://localhost:5000 -u 100 -r 20 -t 5m

# Ultra heavy load with 3 containers
docker-compose up -d --scale api=3
locust -f locustfile.py --host=http://localhost:5000 -u 200 -r 40 -t 10m
```

---

## 📹 Video Demo

**YouTube Link**: [https://youtu.be/ujmNQL2d268]

Video covers:
- ✅ Making single predictions
- ✅ Uploading data and triggering retraining
- ✅ Real-time metrics dashboard
- ✅ Batch predictions
- ✅ API health checks
- ✅ Load testing demonstration

---

## 🎓 Feature Interpretations

### Feature 1: Data Distribution & Class Balance

**Story**: 
- Balanced class distribution across all 12 fruit types
- Training/Validation/Test split: 60% / 20% / 20%
- Each image normalized to 150×150 pixels and [0,1] range
- **Impact**: Ensures fair model training without class bias

### Feature 2: Color Characteristics

**Story**:
- Different fruits have distinct RGB color profiles
- Red fruits (apple, cherry): High red channel
- Yellow fruits (banana, mango): High red+green channels
- Green fruits (kiwi, guava): High green channel
- **Impact**: Color is primary discriminator for fruit classification

### Feature 3: Texture & Edge Complexity

**Story**:
- Smooth fruits (banana, mango): Low edge density
- Textured fruits (strawberry, pineapple): High edge density
- Texture provides complementary features to color
- **Impact**: Robust classification even with poor color information

---

## 🛠️ Troubleshooting

### Issue: Model not loading

```bash
# Check if model file exists
ls -la models/fruit_classifier_light.h5

# Reinstall TensorFlow
pip install --upgrade tensorflow
```

### Issue: API port already in use

```bash
# Find process using port 5000
lsof -i :5000

# Kill process
kill -9 <PID>

# Or use different port
python src/api.py --port 5001
```

### Issue: Dataset not found

```bash
# Verify dataset structure
ls -la data/

# Check if classes exist
ls -la data/apple/
```

### Issue: Out of memory

```bash
# Reduce batch size in training
# Or use a smaller model

# Monitor memory usage
watch -n 1 'docker stats'
```

---

## 📚 References

- [TensorFlow Keras](https://keras.io/)
- [MobileNetV2](https://arxiv.org/abs/1801.04381)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Docker Documentation](https://docs.docker.com/)
- [Locust Documentation](https://docs.locust.io/)

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

Bella Melissa Ineza
---

## 📞 Support

For issues or questions, please open a GitHub issue.

---

## ✅ Checklist for Submission

- [x] Jupyter notebook with full pipeline
- [x] Python scripts (preprocessing, model, prediction, API, retraining)
- [x] Web UI dashboard
- [x] API endpoints
- [x] Docker configuration
- [x] Load testing script
- [x] Model file (.h5)
- [x] README with documentation
- [ ] GitHub repository URL
- [ ] YouTube demo video

---

