# Prodigal AI Batch 5  
## Week 2: 2 May–16 May ’25  
### Team Lead: Ishan Mittal  
### Team Members: Avinash Reddy, Divya Rao, Prem Bagga  

## Model Export & Optimization Formats

### Introduction  
As models move from experimentation to production, it becomes critical to represent them in formats compatible with different deployment environments and to optimize them for efficiency without compromising accuracy. This phase focused on understanding common export formats, their associated challenges, and effective optimization strategies for real-world deployment.

### Export Formats

#### 1. ONNX (Open Neural Network Exchange):  
ONNX provides an interoperable format that allows models trained in PyTorch, TensorFlow, or other frameworks to be ported seamlessly across platforms. It is widely supported by inference runtimes such as ONNX Runtime, TensorRT, and OpenVINO.

#### 2. TorchScript (PyTorch):  
TorchScript supports exporting PyTorch models for deployment by transforming them into static or scripted graphs. It enables inference in non-Python environments like C++ and is highly suited for mobile and embedded deployment.

- **Dynamic vs. Static Graphs:**
  Dynamic (eager) graphs offer flexibility but lower performance, while static graphs enable better optimization at the cost of reduced flexibility.

#### 3. TensorFlow SavedModel:  
The SavedModel format includes both the computation graph and the weights. It is the default for TensorFlow deployment and is supported across TensorFlow Serving, TFLite, and TensorFlow.js
#### Common Pitfalls:
- **Unsupported Operators:** Certain model operations are not convertible between formats and may cause failures during runtime.
- **Dynamic Shape Handling:** Inference engines struggle with variable input dimensions unless explicitly handled.
- **Precision Differences:** Format conversion may alter numeric precision, leading to slight variations in model predictions.

### Optimization Strategies

#### 1. Post-Training Quantization:  
Applied 8-bit quantization to reduce model size and increase inference speed. This method requires minimal retraining and is effective on edge devices.

#### 2. Model Pruning:  
Redundant or less important weights were removed to slim down the network, decreasing computational cost without major accuracy loss.

#### 3. Knowledge Distillation:  
Used a larger, pre-trained *teacher* model to train a smaller *student* model. This resulted in a lighter model with similar predictive performance.

#### Trade-off Evaluation:
Each optimization strategy was evaluated in terms of:
- **Accuracy impact**
- **Inference speed**
- **Memory usage**

Benchmarks showed that while optimization reduces resource demand, careful tuning is required to prevent performance degradation.

### Conclusion  
Effective model export and optimization are essential to ensure portability and performance in production systems. By understanding the strengths and limitations of various export formats and applying targeted optimization techniques like quantization, pruning, and distillation, teams can deliver efficient and scalable ML solutions ready for deployment across diverse hardware environments.
