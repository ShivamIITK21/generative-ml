import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self, input_dims: int, hidden_dims: int):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.mean_net = nn.Sequential(
                nn.Linear(input_dims, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, hidden_dims),
                )
        self.stddev_net= nn.Sequential(
                nn.Linear(input_dims, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, hidden_dims),
                nn.Sigmoid(),
                )

    def forward(self, x: torch.Tensor):
        mean = self.mean_net(x)
        stddev = self.stddev_net(x)
        return (torch.normal(mean, stddev), mean, stddev)

class Decoder(nn.Module):
    def __init__(self, hidden_dims: int, output_dims: int):
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.mean_net = nn.Sequential(
                nn.Linear(hidden_dims, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, output_dims),
                )
        self.stddev_net= nn.Sequential(
                nn.Linear(hidden_dims, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, output_dims),
                nn.Sigmoid(),
                )
        
    def forward(self, z):
        mean = self.mean_net(z)
        stddev = self.stddev_net(z)
        return torch.normal(mean, stddev)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.rand(28, 28).to(device)
    x = x.view(-1, 28*28)
    en = Encoder(28*28, 256).to(device)
    out, _, _= en.forward(x)
    print(out.size())


