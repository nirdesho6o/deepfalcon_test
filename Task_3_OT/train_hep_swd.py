import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset_hep import JetImageDataset
from model_hep import HEPAutoencoder
import numpy as np

def sliced_wasserstein_distance(z, num_projections=50, device='cpu'):
    batch_size, latent_dim = z.size()
    target_z = torch.randn_like(z)
    projections = torch.randn(latent_dim, num_projections, device=device)
    projections = projections / torch.norm(projections, dim=0, keepdim=True)
    
    z_proj = torch.matmul(z, projections)
    target_proj = torch.matmul(target_z, projections)
    
    z_proj_sorted, _ = torch.sort(z_proj, dim=0)
    target_proj_sorted, _ = torch.sort(target_proj, dim=0)
    return torch.mean((z_proj_sorted - target_proj_sorted) ** 2)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

dataset = JetImageDataset("../data/quark-gluon.hdf5", limit=1000)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = HEPAutoencoder(latent_dim=128).to(device)
criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 15
SWD_WEIGHT = 10.0 


print("Training HEP Autoencoder with SWD...")
for epoch in range(EPOCHS):
    model.train()
    total_recon = 0
    total_swd = 0
    
    for images, _ in train_loader:
        images = images.to(device)
        
        reconstructed, latent_vector = model(images)
        recon_loss = criterion(reconstructed, images) / images.size(0)
        swd_loss = sliced_wasserstein_distance(latent_vector, device=device)
        
        loss = recon_loss + (SWD_WEIGHT * swd_loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_recon += recon_loss.item()
        total_swd += swd_loss.item()
        
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Recon Loss: {total_recon/len(train_loader):.4f} | SWD Loss: {total_swd/len(train_loader):.4f}")

torch.save(model.state_dict(), "hep_ae_swd.pt")
print("--> Saved model to 'hep_ae_swd.pt'")


def plot_hep_reconstruction(model, dataloader, num_images=4):
    model.eval()
    images, labels = next(iter(dataloader))
    images = images.to(device)
    
    with torch.no_grad():
        reconstructed, _ = model(images)

    images = images.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()
    
    def normalize_for_plot(img_array):
        img_array = np.clip(img_array, 0, None)
        
        # Scale between 0 and 1
        vmax = img_array.max(axis=(1,2,3), keepdims=True)
        return img_array / (vmax + 1e-8)

    images = normalize_for_plot(images)
    reconstructed = normalize_for_plot(reconstructed)

    fig, axes = plt.subplots(2, num_images, figsize=(12, 6))
    for i in range(num_images):
        orig_img = images[i].transpose(1, 2, 0)
        recon_img = reconstructed[i].transpose(1, 2, 0)
        
        label_str = "Quark" if labels[i].item() == 1 else "Gluon"
        
        axes[0, i].imshow(orig_img)
        axes[0, i].set_title(f"Original {label_str}")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(recon_img)
        axes[1, i].set_title("SWD Reconstructed")
        axes[1, i].axis('off')
        
    plt.tight_layout()
    plt.savefig("hep_reconstruction_swd.png")
    print("--> Saved plot to 'hep_reconstruction_swd.png'")
    plt.show()

plot_hep_reconstruction(model, train_loader)