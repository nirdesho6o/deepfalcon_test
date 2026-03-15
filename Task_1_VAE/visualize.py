import torch
import matplotlib.pyplot as plt

from dataset import JetDataset
from model import VAE


dataset = JetDataset("data/quark-gluon.hdf5")

model = VAE()
model.load_state_dict(torch.load("vae_model.pt"))
model.eval()

x = dataset[0].unsqueeze(0)

# same preprocessing used in training
denom = x.amax(dim=(2,3), keepdim=True)
denom[denom == 0] = 1
x = x / denom
x = x * 10
x = torch.clamp(x,0,1)

with torch.no_grad():
    recon,_,_ = model(x)

print("Recon stats:", recon.min().item(), recon.max().item())

x = x.squeeze().numpy()
recon = recon.squeeze().numpy()

# normalize reconstruction for display
recon = recon / (recon.max() + 1e-8)

fig, axs = plt.subplots(3,2, figsize=(6,8))

titles = ["ECAL","HCAL","Tracks"]

for i in range(3):

    axs[i,0].imshow(x[i], vmin=0, vmax=1)
    axs[i,0].set_title("Original "+titles[i])

    axs[i,1].imshow(recon[i], vmin=0, vmax=1)
    axs[i,1].set_title("Reconstructed "+titles[i])

    axs[i,0].axis("off")
    axs[i,1].axis("off")

plt.tight_layout()
plt.savefig("figures/reconstruction_example.png", dpi=300)
plt.show()