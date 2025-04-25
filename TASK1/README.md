# Hello, Inference!

A simple Python application demonstrating a basic machine learning inference task using PyTorch. The application runs a dummy linear model on random input data, utilizing a GPU (CUDA) if available or falling back to a CPU. This project serves as a starting point for building more complex inference pipelines.

---

## Prerequisites

- **Python**: 3.8 or higher  
- **Conda or Virtualenv**: For managing Python environments  
- **Docker**: For containerized deployment   
- **Git**: For version control  
- **NVIDIA GPU and CUDA**: Required for GPU-based inference 

---

## Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd inference-project
```

### 2.Creating a Python Environment
- **Using Conda**:
```bash
conda create -n inference_env python=3.8
conda activate inference_env
pip install -r requirements.txt
```
- **Using Virtualenv**:
```bash
python -m venv inference_env
source inference_env/bin/activate  # On Windows: inference_env\Scripts\activate
pip install -r requirements.txt
```
### 3.Run the application
```bash
python hello_inference.py
```
- ***Exprected Output***:
```bash
Hello, Inference! Running on <cuda|cpu>. Output: <some_float_value>
```

### 4.Run Unit Tests
```bash
pytest
```

## Docker Deployment

### Build the Docker image:
```bash
docker build -t hello-inference .
```

### Run the container:
```bash
docker run --gpus all hello-inference
```
If no GPU is available, the application will fall back to CPU.

## CI/CD
A GitHub Actions pipeline is configured(see `.github/workflows/docker.yml `) to:
- Run unit tests on every push or pull request
- Build and push the Docker image to Docker Hub on pushes to the `main` branch





