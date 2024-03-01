import torch.nn as nn
import torch
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, hidden_dims: int):
        super().__init__()
        self.l1 = nn.Linear(784, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, 256)
        self.l4 = nn.Linear(256, hidden_dims)
        self.l5 = nn.Linear(256, hidden_dims)
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()

    def forward(self, x: torch.Tensor):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.l1(x)) 
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        mean = self.l4(x)
        sigma= torch.exp(self.l5(x))
        z = mean + sigma*self.N.sample(mean.shape)
        return mean, sigma, z

class Decoder(nn.Module):
    def __init__(self, hidden_dims: int, output_dims: int):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.l1 = nn.Linear(hidden_dims, 512)
        self.l2 = nn.Linear(512, 784)
        
    def forward(self, z):
        z = F.relu(self.l1(z))
        z = F.sigmoid(self.l2(z))
        z = z.reshape(-1, 1, 28, 28)
        return z

def VAE_Loss(x_hat, x, mean, sigma):
    kl = (1 + torch.log(sigma) - mean**2 - sigma**2)
    kl = kl.sum()
    total_loss = ((x-x_hat)**2).sum() - kl
    return total_loss

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.rand(32, 1, 28, 28).to(device)
    encoder = Encoder(2).to(device)
    decoder = Decoder(2, 28*28).to(device)
    encoder_mean, encoder_stddev = encoder(x)
    print(f"encoder_mean = {encoder_mean.size()} encoder_stddev = {encoder_stddev.size()}")
    z = torch.normal(encoder_mean, encoder_stddev)
    print(f"z={z.size()} encoder_mean={encoder_mean.size()} encoder_stddev={encoder_stddev.size()}")
    x_regen = decoder(z)
    print(f"Loss = {VAE_Loss(x, x_regen, encoder_mean, encoder_stddev).size()}")



