import torch
import torch.nn as nn

LATENT_DIM = 32


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3,32,4,2,1),
            nn.ReLU(),
            nn.Conv2d(32,64,4,2,1),
            nn.ReLU(),
            nn.Conv2d(64,128,4,2,1),
            nn.ReLU(),
            nn.Conv2d(128,256,4,2,1),
            nn.ReLU()
        )

        self.flatten = nn.Flatten()

        self.fc_mu = nn.Linear(256*8*8, LATENT_DIM)
        self.fc_logvar = nn.Linear(256*8*8, LATENT_DIM)

    def forward(self,x):

        x = self.conv(x)
        x = self.flatten(x)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(LATENT_DIM,256*8*8)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1),
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,4,2,1),
            nn.ReLU(),
            nn.ConvTranspose2d(64,32,4,2,1),
            nn.ReLU(),
            nn.ConvTranspose2d(32,3,4,2,1),
            nn.Sigmoid()
        )
    def forward(self,z):

        x = self.fc(z)
        x = x.view(-1,256,8,8)

        x = self.deconv(x)

        return x


class VAE(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return mu + eps*std

    def forward(self,x):

        mu, logvar = self.encoder(x)

        z = self.reparameterize(mu,logvar)

        recon = self.decoder(z)

        return recon, mu, logvar