import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 16)
        self.layer2 = nn.Linear(16, 16)
        self.layer3 = nn.Linear(16, 1)
    
    def forward(self, input):
        x = self.layer1(input)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        output = self.layer3(x)
        return output
    
if __name__ == "__main__":
    model = MLP()  

    input_value = torch.tensor([[42.0]])  # shape: (1, 1)

    output = model(input_value)

    print(f"Input: {input_value.item()}, Output: {output.item()}")
