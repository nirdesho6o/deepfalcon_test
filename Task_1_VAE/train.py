import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import h5py
from model import VAE
import numpy as np



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading subset from HDF5...")

with h5py.File("data/quark-gluon.hdf5", "r") as f:
    data = f["X_jets"][:10000]

data = np.transpose(data, (0,3,1,2))
data = np.log1p(data)

data = torch.tensor(data, dtype=torch.float32)

# resize to 128x128
data = F.interpolate(data, size=(128,128), mode="bilinear")

loader = DataLoader(data, batch_size=64, shuffle=True)

model = VAE().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)


BETA = 0.001

def loss_function(recon, x, mu, logvar):

    # weighted reconstruction loss
    weight = 1 + 20 * x
    recon_loss = ((recon - x)**2 * weight).mean()

    # KL divergence
    kl = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    loss = recon_loss + BETA * kl

    return loss


EPOCHS = 10


for epoch in range(EPOCHS):

    total_loss = 0
    printed = False
    loop = tqdm(loader)

    for x in loop:

        x = x.to(device)

        denom = x.amax(dim=(2,3), keepdim=True)
        denom[denom == 0] = 1
        x = x / denom
        x = x * 10
        x = torch.clamp(x, 0, 1)

        if not printed:
            print("Input stats:", x.min().item(), x.max().item(), x.mean().item())
            printed = True

        recon, mu, logvar = model(x)

        loss = loss_function(recon, x, mu, logvar)

        if torch.isnan(loss):
            print("NaN detected, skipping batch")
            continue

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

        loop.set_description(f"Epoch {epoch}")
        loop.set_postfix(loss=loss.item())

    print("avg loss:", total_loss / len(loader))


torch.save(model.state_dict(), "vae_model.pt")