# 🌱 MLOps-TeamBeans

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)
![MLflow](https://img.shields.io/badge/MLflow-2.8.0-blue.svg)
![DVC](https://img.shields.io/badge/DVC-3.0.0+-orange.svg)

**A Production-Ready MLOps Pipeline for Bean Disease Classification**

[Features](#-features) • [Architecture](#-architecture) • [Getting Started](#-getting-started) • [Model](#-model) • [Pipeline](#-pipeline) • [Web App](#-web-application)

</div>

---

## 📋 Overview

An end-to-end MLOps project implementing a **deep learning pipeline** for classifying bean leaf diseases. This project demonstrates best practices in machine learning operations, including experiment tracking, data versioning, validation, model monitoring, and deployment.

The model classifies bean leaf images into three categories:
- 🔴 **Angular Leaf Spot** - Bacterial disease caused by *Pseudomonas syringae pv. lachrymans*
- 🟤 **Bean Rust** - Fungal disease caused by *Uromyces phaseoli typica*
- 🟢 **Healthy** - No disease detected

## ✨ Features

### 🔬 **ML Pipeline**
- **Deep Learning Model**: Custom CNN architecture (2 Conv + 2 FC layers)
- **PyTorch Implementation**: Modern deep learning framework
- **Balanced Dataset**: 1,295 images (500×500 RGB) across 3 classes

### 🛠️ **MLOps Tools & Practices**
- **Experiment Tracking**: MLflow for metrics, parameters, and model versioning
- **Data Version Control**: DVC for reproducible data pipelines
- **Data Validation**: 
  - Great Expectations for schema and quality checks
  - DeepChecks for ML-specific validations
- **Pipeline Automation**: DVC pipelines for end-to-end workflow
- **Carbon Tracking**: CodeCarbon for environmental impact monitoring
- **Testing**: Pytest suite for model and API validation

### 🚀 **Deployment**
- **Web Application**: Flask-based interface for predictions
- **REST API**: FastAPI backend for model serving
- **Containerization**: Docker support for frontend and backend
- **Notebooks**: Jupyter notebooks for exploratory analysis

## 🏗️ Architecture

```
┌─────────────────┐
│  Data Source    │
│  (HuggingFace)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Validation │
│ Great Expect.   │
│ DeepChecks      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│ Data Processing │◄─────┤     DVC      │
│   DataLoaders   │      │   Pipeline   │
└────────┬────────┘      └──────────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│ Model Training  │◄────►│    MLflow    │
│   PyTorch CNN   │      │   Tracking   │
└────────┬────────┘      └──────────────┘
         │
         ▼
┌─────────────────┐
│    Evaluation   │
│ Test Metrics    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│       Deployment            │
│  ┌──────────┐  ┌─────────┐ │
│  │ FastAPI  │  │  Flask  │ │
│  │   API    │  │  Web UI │ │
│  └──────────┘  └─────────┘ │
└─────────────────────────────┘
```

## 📁 Project Structure

```
MLOps-TeamBeans/
├── src/
│   ├── data/              # Data loading and validation
│   │   ├── load_data.py
│   │   ├── make_dataset.py
│   │   └── deepchecks_validations.py
│   ├── features/          # Feature engineering and validation
│   │   ├── build_features.py
│   │   └── gx/            # Great Expectations configs
│   ├── models/            # Model training and evaluation
│   │   ├── model.py
│   │   ├── train_model.py
│   │   └── test_model.py
│   ├── app/               # FastAPI application
│   │   └── api.py
│   └── web/               # Flask web interface
│       ├── app.py
│       └── templates/
├── notebooks/             # Jupyter notebooks for analysis
├── tests/                 # Test suite
├── mlruns/               # MLflow experiment tracking
├── dvc.yaml              # DVC pipeline definition
└── requirements.txt      # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip or conda
- Git
- (Optional) Docker for containerized deployment

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/MLOps-essi-upc/MLOps-TeamBeans.git
cd MLOps-TeamBeans
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Pull data with DVC**
```bash
dvc pull
```

### 🏃 Quick Start

#### Run the Complete Pipeline

```bash
dvc repro
```

This executes the full pipeline:
1. **make_dataset**: Process raw data into PyTorch dataloaders
2. **train**: Train the CNN model with MLflow tracking
3. **evaluate**: Test model performance and generate metrics

#### Start MLflow UI

```bash
mlflow ui
```
Access at `http://localhost:5000` to view experiments, metrics, and model artifacts.

#### Run the Web Application

```bash
python src/web/app.py
```
Access at `http://localhost:5001` to upload images and get predictions.

#### Start the FastAPI Server

```bash
uvicorn src.app.api:app --reload
```
API available at `http://localhost:8000` with interactive docs at `/docs`.

## 🤖 Model

### Architecture: BeansTeam 2CL+2FL

- **Input**: 500×500×3 RGB images
- **Layers**:
  - 2 Convolutional layers (2D) with ReLU activation
  - 2 Fully connected layers
  - LogSoftmax output layer (3 neurons)
- **Framework**: PyTorch 2.1.0
- **Loss Function**: Negative Log-Likelihood Loss
- **Optimizer**: Adam

### Training Details

- **Dataset Split**:
  - Training: 1,034 images
  - Validation: 133 images
  - Test: 128 images
- **Monitoring**: Carbon emissions tracked with CodeCarbon
- **Experiment Tracking**: All runs logged to MLflow

See [BeansTeam 2CL+2FL_model_card.md](BeansTeam%202CL+2FL_model_card.md) for complete model documentation.

## 🔄 Pipeline

The project uses **DVC** for pipeline orchestration:

```yaml
Stages:
  make_dataset → train → evaluate
```

Each stage:
- Tracks dependencies (data, code)
- Generates versioned outputs
- Enables reproducibility

**Run individual stages:**
```bash
dvc repro make_dataset
dvc repro train
dvc repro evaluate
```

## 📊 Data

### Dataset Details

- **Source**: HuggingFace Beans Dataset
- **Size**: 1,295 images
- **Format**: 500×500 RGB JPEG
- **Balance**: Evenly distributed across 3 classes
- **License**: MIT

### Data Validation

**Great Expectations**: Schema validation, statistical checks, and data quality assertions
```bash
great_expectations checkpoint run checkpoint_train
```

**DeepChecks**: ML-specific validations including distribution checks and label integrity

## 🧪 Testing

```bash
pytest tests/
```

Test coverage includes:
- Model architecture and forward pass
- API endpoints (FastAPI & Flask)
- MLflow logging functionality
- Training pipeline components

## 🐳 Docker Deployment

### Backend (FastAPI)
```bash
docker build -f Dockerfile-backend.txt -t beans-backend .
docker run -p 8000:8000 beans-backend
```

### Frontend (Flask)
```bash
docker build -f Dockerfile-frontend.txt -t beans-frontend .
docker run -p 5001:5001 beans-frontend
```

## 🌐 Web Application

The Flask web interface provides:
- Image upload functionality
- Real-time disease prediction
- Confidence scores for each class
- User-friendly visualization of results

## 📈 Monitoring & Tracking

- **MLflow**: Tracks experiments, parameters, metrics, and model artifacts
- **CodeCarbon**: Monitors CO₂ emissions during training
- **Metrics**: Accuracy, loss, confusion matrix stored in `models/test_metrics.csv`

## 🔗 Links
- **Dataset**: [Beans on HuggingFace](https://huggingface.co/datasets/beans)

---

<div align="center">

</div>
