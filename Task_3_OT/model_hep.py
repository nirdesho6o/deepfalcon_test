import torch
import torch.nn as nn
import torch.nn.functional as F

class HEPAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        
        
        self.enc_conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc_conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
      
        self.flatten_size = 128 * 7 * 7
        self.fc_enc = nn.Linear(self.flatten_size, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, self.flatten_size)
        
        self.dec_conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.dec_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec_conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.dec_conv4 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def encode(self, x):
        x = F.max_pool2d(F.relu(self.enc_conv1(x)), 2) # -> 62x62
        x = F.max_pool2d(F.relu(self.enc_conv2(x)), 2) # -> 31x31
        x = F.max_pool2d(F.relu(self.enc_conv3(x)), 2) # -> 15x15
        x = F.max_pool2d(F.relu(self.enc_conv4(x)), 2) # -> 7x7
        x = x.view(-1, self.flatten_size)
        return self.fc_enc(x)

    def decode(self, z):
        x = F.relu(self.fc_dec(z))
        x = x.view(-1, 128, 7, 7)
        
        
        x = F.interpolate(x, size=(15, 15), mode='nearest')
        x = F.relu(self.dec_conv1(x))
        
        x = F.interpolate(x, size=(31, 31), mode='nearest')
        x = F.relu(self.dec_conv2(x))
        
        x = F.interpolate(x, size=(62, 62), mode='nearest')
        x = F.relu(self.dec_conv3(x))
        
        x = F.interpolate(x, size=(125, 125), mode='nearest')
       
        x = self.dec_conv4(x)  # Removed F.relu()
        return x

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

class HEPLatentGenerator(nn.Module):
    def __init__(self, noise_dim=64, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, noise):
        return self.net(noise)