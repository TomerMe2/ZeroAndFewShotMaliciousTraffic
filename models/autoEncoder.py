import torch


class AutoEncoder(torch.nn.Module):
    
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        
        self.model = torch.nn.Sequential(
          torch.nn.Linear(input_size, input_size//2),
          torch.nn.ReLU(),
          torch.nn.Linear(input_size//2, input_size//4),
          torch.nn.ReLU(),
          torch.nn.Linear(input_size//4, input_size//10),
          torch.nn.ReLU(),
          
          torch.nn.Linear(input_size//10, input_size//4),
          torch.nn.ReLU(),
          torch.nn.Linear(input_size//4, input_size//2),
          torch.nn.ReLU(),
          torch.nn.Linear(input_size//2, input_size),
        )

    def forward(self, x):
        return self.model(x)