import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return torch.relu(self.fc(x))

model = SimpleModel()
model.eval()
example_input = torch.randn(1, 4)

traced_model = torch.jit.trace(model, example_input)
traced_model.save("traced_model.pt")

scripted_model = torch.jit.script(model)
scripted_model.save("scripted_model.pt")

with torch.no_grad():
    original_output = model(example_input)
    traced_output = traced_model(example_input)
    scripted_output = scripted_model(example_input)

print("Original:", original_output)
print("Traced:", traced_output)
print("Scripted:", scripted_output)
