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
        return mean, stddev 

class Decoder(nn.Module):
    def __init__(self, hidden_dims: int, output_dims: int):
        super().__init__()
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
        return mean, stddev

def VAE_Loss(input_image, decoder_mean, decoder_stddev, encoder_mean, encoder_stddev):
    x_z_dist = torch.distributions.normal.Normal(decoder_mean, decoder_stddev)

    encoder_mean_sq = torch.square(encoder_mean)
    encoder_stddev_sq = torch.square(encoder_stddev)
    encoder_stddev_sq_log = torch.log(encoder_stddev_sq)
    encoder_loss_tensor = encoder_stddev_sq_log - encoder_mean_sq - encoder_stddev_sq 
    print(f"encoder_loss_tensor_size = {encoder_loss_tensor.size()}")
    encoder_loss = torch.sum(encoder_loss_tensor, dim=1)
    print(f"encoder_loss = {encoder_loss}, encoder_loss size= {encoder_loss.size()}")

    log_prob_tensor = x_z_dist.log_prob(input_image)
    decoder_loss = torch.sum(log_prob_tensor, dim=1)
    print(f"log_prob ={decoder_loss}")
    total_loss = -(encoder_loss+decoder_loss)
    print(f"total loss={total_loss}")
    return total_loss

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.rand(28, 28).to(device)
    x = x.view(-1, 28*28)
    encoder = Encoder(28*28, 256).to(device)
    decoder = Decoder(256, 28*28).to(device)
    encoder_mean, encoder_stddev = encoder(x)
    z = torch.normal(encoder_mean, encoder_stddev)
    print(f"z={z.size()} encoder_mean={encoder_mean.size()} encoder_stddev={encoder_stddev.size()}")
    decoder_mean, decoder_stddev = decoder(z)
    print(f"decoder_mean={decoder_mean.size()} decoder_stddev={decoder_stddev.size()}")
    VAE_Loss(x, decoder_mean, decoder_stddev, encoder_mean, encoder_stddev)



