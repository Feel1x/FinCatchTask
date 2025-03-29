import torch
import torch.nn as nn

"""
MLP for simulating a basic LLM analysis.
"""

class MLP(nn.Module):
    def __init__(self):
        # define layers: layer1 (1->16), layer2 (16->1)
        super().__init__()
        self.layer1 = nn.Linear(1, 16)
        self.layer2 = nn.Linear(16, 1)
    
    def forward(self, x):
        # process input through hidden layer & compute output
        x = self.layer1(x)
        x = torch.relu(x) # Negative => 0 
        x = self.layer2(x)
        return x
        
    def analyze(self, market_indicator: int) -> int:
         # if 1-9, return digit repeated 4x; else, process indicator and scale output
        if 1 <= market_indicator <= 9:
            return int(str(market_indicator) * 4)
        
        x = torch.tensor([[market_indicator]], dtype=torch.float32)
        output = self.forward(x)
        
        #sigmoid to squash output, then adjust to [1, 10000]
        sigmoid_out = torch.sigmoid(output).item() 
        scaled = (sigmoid_out ** 0.5) * (10000 - 1) + 1 
    
        return int(scaled)

    
if __name__ == "__main__":
    mlp = MLP()
    print(mlp.analyze(3))   # output -> 3333
    print(mlp.analyze(55))  # output -> random
