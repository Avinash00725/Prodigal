# Hello, Inference!

A simple Python application demonstrating a basic machine learning inference task using PyTorch. The application runs a dummy linear model on random input data, utilizing a GPU (CUDA) if available or falling back to a CPU. This project serves as a starting point for building more complex inference pipelines.

---

## Prerequisites

- **Python**: 3.8 or higher  
- **Conda or Virtualenv**: For managing Python environments  
- **Docker**: For containerized deployment (optional)  
- **Git**: For version control  
- **NVIDIA GPU and CUDA**: Required for GPU-based inference (optional)

---

## Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd inference-project

### 2.Creating a Python Environment
- **Using Conda**:
```bash
conda create -n inference_env python=3.8
conda activate inference_env
pip install -r requirements.txt

