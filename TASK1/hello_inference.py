import torch
import numpy as np

def run_inference():
    # Dummy model: Linear layer
    model = torch.nn.Linear(10, 1)
    input_data = torch.tensor(np.random.rand(1, 10), dtype=torch.float32)
    # Use CUDA if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_data = input_data.to(device)
    output = model(input_data)
    print(f"Hello, Inference! Running on {device}. Output: {output.item()}")

if __name__ == "__main__":
    run_inference()
