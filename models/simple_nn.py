import torch


class SimpleNN(torch.nn.Module):
    
    def __init__(self, input_size):
        super().__init__()
        self.embedding_size = 512
        self.model = torch.nn.Sequential(
          torch.nn.Linear(input_size, 128),
          torch.nn.ReLU(),
          torch.nn.Linear(128, 256),
          torch.nn.ReLU(),
          torch.nn.Linear(256, self.embedding_size),
        )

    def forward(self, x):
        return self.model(x)