import torch
import torch.nn.functional as F 
from torch import nn 

# Input img --> Hidden dim --> mean, std --> Re-parametrization trick --> Decoder --> Output img
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=20):
        super().__init__()
        # encoder
        self.img_2hid = nn.Linear(input_dim, h_dim)
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2sigma = nn.Linear(h_dim, z_dim)
        
        # decoder
        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)

        self.relu = nn.ReLU()

    
    def encode(self, x):
        # q_phi(z|x)
        h = self.relu(self.img_2hid(x))
        mu = self.hid_2mu(h)
        sigma = self.hid_2sigma(h)
        return mu, sigma

    def decode(self, z):
        # p_theta(x|z)
        h = self.relu(self.z_2hid(z))
        return torch.sigmoid(self.hid_2img(h)) # Enforce [0, 1], pixel values for MNIST

    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_reparam = mu + sigma*epsilon
        x_reconstructed = self.decode(z_reparam)
        return x_reconstructed, mu, sigma

if __name__ == "__main__":
    x = torch.randn(4, 784) # 28x28 = 784, MNIST dimensions, batch size of 4
    vae = VariationalAutoEncoder(input_dim=784)
    x_reconstructed, mu, sigma = vae(x)
    print(x_reconstructed.shape)
    print(mu.shape)
    print(sigma.shape)