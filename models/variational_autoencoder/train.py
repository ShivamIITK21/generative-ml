import torch
import torchvision
from variational_autoencoder.model import Encoder, Decoder
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 1e-4
    z_dim = 256
    image_dim = 28*28
    batch_size = 32
    epocs = 50
    
    encoder = Encoder(image_dim, z_dim).to(device)
    decoder = Decoder(z_dim, image_dim).to(device)

    dataset = datasets.MNIST(root="./data", transform=torchvision.transforms.ToTensor(), download=True) 
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 

    opt_encoder = optim.Adam(encoder.parameters(), lr = lr) 
    opt_decoder = optim.Adam(decoder.parameters(), lr = lr) 


