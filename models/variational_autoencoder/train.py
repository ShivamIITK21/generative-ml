import torch
import torchvision
from model import Encoder, Decoder, VAE_Loss
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 1e-3
    z_dim = 2
    image_dim = 28*28
    batch_size = 64 
    epochs = 50
    
    encoder = Encoder(z_dim).to(device)
    decoder = Decoder(z_dim, image_dim).to(device)

    transforms = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor()]
    )
    dataset = datasets.MNIST(root="./data", transform=transforms, download=True) 
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 

    opt_encoder = optim.Adam(encoder.parameters(), lr = lr) 
    opt_decoder = optim.Adam(decoder.parameters(), lr = lr) 

    writer_real = SummaryWriter(f"logs/real")
    writer_regen = SummaryWriter(f"logs/regen")

    encoder.train()
    decoder.train()
    step = 0
    for epoch in range(epochs):
        for batch_idx, (x, _) in enumerate(loader):
            x = x.to(device)
            encoder_mean, encoder_stddev, z = encoder(x)
            x_regen = decoder(z)
            loss = VAE_Loss(x, x_regen, encoder_mean, encoder_stddev)
            encoder.zero_grad()
            decoder.zero_grad()
            loss.backward()
            opt_encoder.step()
            opt_decoder.step()

            if batch_idx == 0:
                print(
                    f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(loader)} \
                        Loss : {loss}"
                )

                with torch.no_grad():
                    image_grid_real = torchvision.utils.make_grid(x)
                    image_grid_regen = torchvision.utils.make_grid(x_regen)
                    
                    writer_real.add_image("Real images", image_grid_real, global_step=step)
                    writer_regen.add_image("Regenerated images", image_grid_regen, global_step=step)
                    step += 1
            
    encoder_scripted = torch.jit.script(encoder)
    decoder_scripted = torch.jit.script(decoder)
    encoder_scripted.save("encoder_mnist_basic.pt")
    decoder_scripted.save("decoder_mnist_basic.pt")
