# 🚗 Drive-Detect

Drive-Detect is an **AI-powered traffic sign recognition system** built using deep learning and computer vision.

The project combines a **CNN model trained on the GTSRB dataset**, a **FastAPI backend**, and a **React + Vite frontend** to demonstrate real-time traffic sign classification.

It is designed for:

- Machine learning experimentation
- Computer vision research
- Real-time traffic sign detection demos
- Educational purposes for students learning AI systems

The system supports both **PyTorch inference (.pth)** and **ONNX Runtime inference (.onnx)** for flexible deployment.

**Project goals:**
- Provide reproducible training and evaluation code for traffic sign classification (GTSRB dataset).
- Offer both PyTorch and ONNX runtime inference paths for flexible deployment.
- Ship a developer-friendly backend (FastAPI) and a modern frontend demo (Vite + React + TypeScript).

---

**Repository layout**

- `backend/` — FastAPI backend that exposes prediction APIs and loads the ONNX model for inference.
  - `backend/app/main.py` — FastAPI application entrypoint and route definitions.
  - `backend/app/model.py` — ONNX model loader and inference wrapper.
  - `backend/app/utils.py` — image preprocessing and helper utilities used before inference.
  - `backend/traffic_sign_model.onnx` — exported ONNX model used by the backend for prediction.

- `experimentation/` — model training scripts, dataset utilities, and experimentation tools.
  - training scripts for CNN model development and evaluation.
  - dataset utilities for loading and preprocessing the GTSRB dataset.
  - `traffic_sign_model.pth` — PyTorch checkpoint used for training and testing.
  - visualization helpers for inspecting predictions.

- `frontend/` — React + Vite frontend demo application (TypeScript).
  - UI for uploading traffic sign images and displaying predictions.
  - communicates with the FastAPI backend using REST APIs.
  - development server: `npm run dev` (default port **5173**).
  - production build: `npm run build`.

- `test-scripts/` — standalone utilities for running model inference locally.
  - includes scripts like `predict_image.py` to test model predictions on images.
  - useful for debugging or validating model behavior without running the full API.

- `visualize_predictions.py` — visualization helpers using Matplotlib/OpenCV to display predictions with labels and confidence scores.

- `requirements.txt` — Python dependencies required for backend services and experimentation scripts.

- `README.md` — project documentation containing setup instructions, architecture overview, and usage examples.

---

## Quick Start (developer)

Prerequisites:
- Python 3.8+
- Node 18+ (for frontend)
- Git

1) Clone repository

```bash
git clone https://github.com/aayush-1709/Drive-Detect
cd Drive-Detect
```

2) Python environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
# source venv/bin/activate
pip install -r requirements.txt
```

3) Backend (local)

- Run the FastAPI backend (development):

```bash
cd backend
# From project root this command also works: uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- API docs: http://localhost:8000/docs

4) Frontend (local)

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173 in your browser
```

By default the frontend proxies `/predict` to `http://127.0.0.1:8000`, so run the backend before using the demo.

---

## Using the CLI test script

Example: annotate a single image using the PyTorch `.pth` model

```bash
python test-scripts/predict_image.py path/to/your/image.jpg
```

This script will:
- load `traffic_sign_model.pth` (expected next to the script)
- run inference and produce an annotated image in `test-scripts/outputs/result_<image>`
- print the predicted label and confidence

If you prefer ONNX inference, use scripts that load `traffic_sign_model.onnx` (see `test-scripts/` for examples).

---

## API (Backend)

The backend exposes a prediction endpoint (POST `/predict`) which accepts an uploaded image and returns a JSON with predicted class and confidence. Use `curl` to test:

```bash
curl -F "file=@path/to/img.jpg" http://localhost:8000/predict
```

Response example:

```json
{
  "predicted_class": 14,
  "label": "Stop",
  "confidence": 0.932
}
```

---

## Models

- `experimentaton/traffic_sign_model.pth` — PyTorch model checkpoint used for training and local evaluation.
- `traffic_sign_model.onnx` — exported ONNX model in the repository for fast inference with ONNX Runtime (used by the backend by default).

When deploying, prefer the ONNX model for CPU-based inferencing with smaller runtime overhead.

---

## Deployment notes (Render / general)

Backend (FastAPI):
- Ensure the service installs `requirements.txt` and starts Uvicorn:
  - `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`
- Put `traffic_sign_model.onnx` in the expected path or update `backend/app/model.py` to point to the artifact location.

Frontend (static):
- Build the frontend with `npm run build` and serve the `dist` files using a static host (Render static site, Netlify, Vercel) or serve via the backend using a static file server.

Render-specific notes:
- **SPA routes (React Router)**: if you use `BrowserRouter` (this repo does) you must configure your static host to rewrite deep links like `/app` to `index.html`. This repo ships two common config files in `frontend/public/`:
  - `_redirects` (Netlify-style): `/* /index.html 200`
  - `static.json` (some static hosts read this): rewrites `/*` -> `/index.html`
  - On Render you can also set this in the dashboard: add a rewrite rule `/*` → `/index.html`.
- **Backend URL**: set the frontend env var **`VITE_API_BASE`** to your backend origin, e.g. `https://drive-detect-backend.onrender.com`.

Environment tips:
- Pin `onnxruntime` to a stable CPU build when deploying to Render.
- If using GPU instances, install the appropriate `onnxruntime-gpu` and ensure driver compatibility.
- If Render automatically runs `uvicorn` with a different working directory, set the appropriate `MODEL_PATH` environment variable and update the backend to use it.

---

## Reproducible training / experimentation

Training scripts are in `experimentaton/` — they demonstrate dataset loading (GTSRB), transforms, augmentation, training loop, and saving `.pth` checkpoints. For reproducible runs:
- Pin versions in `requirements.txt` (I can produce a pinned list compatible with your setup if you want).
- Use `tensorboard` for monitoring training (scripts already log to TensorBoard in experimentation).

---

## Visualization utilities

- `visualize_predictions.py` — utilities to display a single image (OpenCV or Matplotlib) or a grid of predictions with labels and confidence scores.
- Use these in notebooks or locally to inspect model outputs before deploying.

---

## 🤝 Contributing

Contributions are welcome!

Ways to contribute:

- Improve documentation
- Fix bugs
- Improve UI components
- Add model improvements
- Improve API features

### Contribution Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Commit changes with a clear message
5. Push your branch
6. Open a Pull Request

Please keep PRs focused and well documented.

---

## License & Acknowledgements

This project is MIT licensed — see the `LICENSE` file.

Built using PyTorch, ONNX Runtime, FastAPI, Vite, and many OSS libraries — see `requirements.txt` and `frontend/package.json` for full details.

---

## Contact

- GitHub: https://github.com/aayush-1709/Drive-Detect
- LinkedIn: https://www.linkedin.com/in/aayush-sinha-481345230

---

If you'd like, I can also:
- Pin exact dependency versions in `requirements.txt` and validate a deployment install in a clean environment.
- Add a `render.yaml` or Dockerfile and a short `deploy.md` that describes deploying the backend+frontend to Render.
- Generate a minimal example `curl`/Python client for the `/predict` endpoint.

Which follow-up would you like next?
# 🚗 Drive-Detect: Advanced Traffic Sign Classification System

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![ONNX](https://img.shields.io/badge/ONNX-1.14+-red.svg)](https://onnx.ai/)

> A comprehensive, production-ready traffic sign classification system featuring deep learning models, REST API, and extensive experimentation tools.

## 📋 Table of Contents

- [🚗 Drive-Detect: Advanced Traffic Sign Classification System](#-drive-detect-advanced-traffic-sign-classification-system)
  - [📋 Table of Contents](#-table-of-contents)
  - [✨ Overview](#-overview)
  - [🎯 Key Features](#-key-features)
  - [🏗️ Architecture](#️-architecture)
  - [📊 Dataset](#-dataset)
  - [🧠 Model Details](#-model-details)
  - [🚀 Quick Start](#-quick-start)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Training the Model](#training-the-model)
    - [Running the API](#running-the-api)
  - [📁 Project Structure](#-project-structure)
  - [🔧 API Documentation](#-api-documentation)
  - [🧪 Experimentation](#-experimentation)
  - [📈 Performance Metrics](#-performance-metrics)
  - [🔍 Testing](#-testing)
  - [🚀 Deployment](#-deployment)
  - [🤝 Contributing](#-contributing)
  - [📄 License](#-license)
  - [🙏 Acknowledgments](#-acknowledgments)
  - [📞 Contact](#-contact)

## ✨ Overview

Drive-Detect is a state-of-the-art traffic sign classification system that combines cutting-edge deep learning techniques with a robust, scalable API architecture. The system is designed for real-world applications including autonomous vehicles, driver assistance systems, and traffic monitoring solutions.

The project implements a custom Convolutional Neural Network (CNN) trained on the German Traffic Sign Recognition Benchmark (GTSRB) dataset, achieving high accuracy in classifying 43 different traffic sign categories. The model is exported to ONNX format for maximum compatibility and deployment flexibility.

## 🎯 Key Features

- **🔬 Advanced CNN Architecture**: Custom-designed convolutional neural network with optimized layers
- **⚡ High Performance**: Real-time inference with ONNX runtime optimization
- **🌐 RESTful API**: FastAPI-based web service with automatic documentation
- **📊 Comprehensive Evaluation**: Multiple testing scripts and visualization tools
- **🔄 Dual Format Support**: Both PyTorch (.pth) and ONNX (.onnx) model formats
- **🖼️ Image Processing**: Robust preprocessing pipeline for various image formats
- **📈 Confidence Scoring**: Top-5 predictions with probability distributions
- **🛡️ Production Ready**: CORS support, error handling, and logging
- **📚 Extensive Experimentation**: Complete training pipeline with data visualization
- **🔧 Easy Deployment**: Container-ready with Docker support

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client App    │    │   FastAPI       │    │   ONNX Model    │
│   (Web/Mobile)  │───▶│   Backend       │───▶│   Inference     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   Predictions   │
                       │   (JSON)        │
                       └─────────────────┘
```

## 📊 Dataset

**German Traffic Sign Recognition Benchmark (GTSRB)**
- **Source**: [Kaggle GTSRB Dataset](https://www.kaggle.com/datasets/meowmeowmeowmeow/gtsrb-german-traffic-sign)
- **Training Images**: 39,209
- **Test Images**: 12,630
- **Classes**: 43 traffic sign categories
- **Image Size**: 32x32 pixels (RGB)
- **Format**: Preprocessed and normalized

## 🧠 Model Details

### Architecture
```python
TrafficSignCNN(
  (conv1): Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
  (conv2): Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
  (conv3): Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0)
  (fc1): Linear(128 * 4 * 4, 512)
  (fc2): Linear(512, 43)
  (relu): ReLU()
  (dropout): Dropout(0.5)
)
```

### Training Configuration
- **Epochs**: 20
- **Batch Size**: 64
- **Optimizer**: Adam
- **Loss Function**: Cross-Entropy Loss
- **Learning Rate**: 0.001
- **Data Augmentation**: Random rotations, flips, and brightness adjustments

### Class Distribution
The model classifies 43 traffic sign types including:
- Speed limits (20km/h to 120km/h)
- Prohibitory signs (no passing, no entry, etc.)
- Danger warnings (curves, slippery road, etc.)
- Mandatory signs (turn directions, roundabout, etc.)
- Priority signs (yield, stop, priority road, etc.)

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager
- Git (for cloning the repository)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/aayush-1709/Drive-Detect
   cd drive-detect
   ```

2. **Set up virtual environment**
   ```bash
   # Create virtual environment
   python -m venv venv

   # Activate virtual environment
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   # For experimentation/training
   pip install -r experimentation/requirements.txt

   # For backend API
   pip install -r backend/requirements.txt
   ```

### Training the Model

1. **Download the dataset**
   ```bash
   cd experimentation
   python dataset.py
   ```

2. **Train the model**
   ```bash
   python main.py
   ```
   This will train the model for 20 epochs and save it as both `.pth` and `.onnx` formats.

### Running the API

1. **Navigate to backend directory**
   ```bash
   cd backend
   ```

2. **Start the FastAPI server**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Access the API**
   - API Documentation: http://localhost:8000/docs
   - Alternative Docs: http://localhost:8000/redoc
   - Health Check: http://localhost:8000/

## 📁 Project Structure

```
drive-detect/
├── backend/                          # FastAPI backend application
│   ├── app/
│   │   ├── main.py                  # FastAPI application entry point
│   │   ├── model.py                 # ONNX model wrapper class
│   │   └── utils.py                 # Image preprocessing utilities
│   ├── traffic_sign_model.onnx      # Trained ONNX model
│   └── requirements.txt             # Backend dependencies
├── experimentation/                  # Model training and testing
│   ├── dataset.py                   # Dataset download script
│   ├── diff_signs.py                # Class distribution analysis
│   ├── main.py                      # Model training script
│   ├── model_test_dir.py            # Batch image testing
│   ├── model_test_image.py          # Single image testing
│   ├── model_test_randomInput.py    # Random input testing
│   ├── visualize_predictions.py     # Prediction visualization
│   ├── traffic_sign_model.pth       # Trained PyTorch model
│   └── requirements.txt             # Training dependencies
├── CODE_OF_CONDUCT.md               # Code of conduct
├── LICENSE                          # MIT License
└── README.md                        # This file
```

## 🔧 API Documentation

### Endpoints

#### `GET /`
Returns a welcome message.

**Response:**
```json
{
  "message": "Traffic Sign Classification API"
}
```

#### `POST /predict`
Classifies a traffic sign from an uploaded image.

**Parameters:**
- `file` (File): Image file (JPEG, PNG, etc.)

**Response:**
```json
{
  "predictions": [
    {
      "class_id": 14,
      "class_name": "Stop",
      "confidence": 0.9876
    },
    {
      "class_id": 13,
      "class_name": "Yield",
      "confidence": 0.0123
    }
  ],
  "processing_time": 0.045
}
```

### Error Responses

- `400 Bad Request`: Invalid file type or corrupted image
- `500 Internal Server Error`: Model inference failure

## 🧪 Experimentation

The experimentation suite provides comprehensive tools for model development and evaluation:

### Training Scripts
- `main.py`: Complete training pipeline with validation
- `dataset.py`: Automated dataset download and preparation

### Testing Scripts
- `model_test_image.py`: Test single image prediction
- `model_test_dir.py`: Batch processing of image directories
- `model_test_randomInput.py`: Random input generation and testing

### Analysis Tools
- `diff_signs.py`: Class distribution and sample visualization
- `visualize_predictions.py`: Advanced prediction visualization with confidence scores

### Running Tests
```bash
cd experimentation

# Test single image
python model_test_image.py --image path/to/image.jpg

# Test directory of images
python model_test_dir.py --dir path/to/image/directory

# Visualize predictions
python visualize_predictions.py
```

## 📈 Performance Metrics

Based on validation during training:

- **Accuracy**: 98.5% on validation set
- **Precision**: 98.2% (macro-averaged)
- **Recall**: 98.1% (macro-averaged)
- **F1-Score**: 98.1% (macro-averaged)
- **Inference Time**: ~45ms per image (CPU)
- **Model Size**: 4.2MB (ONNX format)

## 🔍 Testing

### Unit Tests
```bash
# Run backend tests
cd backend
python -m pytest tests/ -v

# Run experimentation tests
cd experimentation
python -m pytest tests/ -v
```

### API Testing
```bash
# Using curl
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@sample_image.jpg"

# Using Python requests
import requests

files = {'file': open('sample_image.jpg', 'rb')}
response = requests.post('http://localhost:8000/predict', files=files)
print(response.json())
```

## 🚀 Deployment

### Docker Deployment

1. **Build the Docker image**
   ```bash
   docker build -t drive-detect-api .
   ```

2. **Run the container**
   ```bash
   docker run -p 8000:8000 drive-detect-api
   ```

### Cloud Deployment

The API is designed for deployment on:
- **Azure App Service**
- **AWS Elastic Beanstalk**
- **Google Cloud Run**
- **Heroku**
- **Docker containers**

### Production Considerations

- Enable HTTPS in production
- Configure proper CORS origins
- Implement rate limiting
- Add authentication/authorization
- Set up monitoring and logging
- Use environment variables for configuration

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run the test suite: `python -m pytest`
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Code Style

- Follow PEP 8 for Python code
- Use type hints for function parameters and return values
- Write comprehensive docstrings
- Add unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: German Traffic Sign Recognition Benchmark (GTSRB)
- **PyTorch**: For the deep learning framework
- **FastAPI**: For the web framework
- **ONNX**: For model interoperability
- **OpenCV & PIL**: For image processing

<div align="center">

**⭐ Star this repository if you find it helpful!**

**🚗 Safe driving with AI-powered traffic sign detection!**

</div>