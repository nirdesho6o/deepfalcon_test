import torch
import torch.nn as nn
import torch.nn.functional as F
import ot

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- simple autoencoder ----------
class Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784,256),
            nn.ReLU(),
            nn.Linear(256,16)
        )

    def forward(self,x):
        return self.net(x)


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16,256),
            nn.ReLU(),
            nn.Linear(256,784),
            nn.Sigmoid()
        )

    def forward(self,z):
        x = self.net(z)
        return x.view(-1,1,28,28)


encoder = Encoder().to(device)
decoder = Decoder().to(device)

optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=1e-3
)


# ---------- dataset ----------
transform = transforms.ToTensor()

dataset = datasets.MNIST(
    "./data", train=True, download=True, transform=transform
)

loader = DataLoader(dataset, batch_size=128, shuffle=True)


# ---------- training ----------
for epoch in range(10):

    total_loss = 0

    for x,_ in loader:

        x = x.to(device)

        z = encoder(x)
        recon = decoder(z)

        # reconstruction loss
        recon_loss = F.mse_loss(recon, x)

        # sample from standard Gaussian
        z_prior = torch.randn_like(z)

        # compute OT distance
        M = torch.cdist(z, z_prior)**2
        a = ot.unif(z.shape[0])
        b = ot.unif(z.shape[0])

        ot_loss = ot.sinkhorn2(a, b, M.detach().cpu().numpy(), reg=0.1)

        loss = recon_loss + 0.1 * torch.tensor(ot_loss, device=device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("epoch", epoch, "loss", total_loss/len(loader))