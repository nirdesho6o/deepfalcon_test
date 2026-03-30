import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset_mnist import get_mnist_subset
from model_mnist import MNISTAutoencoder


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

# Choosing digits 0 and 4 as suggested by the task
print("Loading MNIST subset...")
train_loader = get_mnist_subset(digit1=0, digit2=4, batch_size=64)

model = MNISTAutoencoder(latent_dim=16).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 10
SWD_WEIGHT = 10.0  


print("SWD Regularized")
for epoch in range(EPOCHS):
    model.train()
    total_recon = 0
    total_swd = 0
    
    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.to(device)

        reconstructed, latent_vector = model(images)
        
        # Calculate both losses
        recon_loss = criterion(reconstructed, images)
        swd_loss = sliced_wasserstein_distance(latent_vector, device=device)
        
        # Combined objective
        loss = recon_loss + (SWD_WEIGHT * swd_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_recon += recon_loss.item()
        total_swd += swd_loss.item()
        
    avg_recon = total_recon / len(train_loader)
    avg_swd = total_swd / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Recon Loss: {avg_recon:.4f} | SWD Loss: {avg_swd:.4f}")

def plot_reconstruction(model, dataloader, digit1=0, digit2=4, num_images=6):
    model.eval()

    dataiter = iter(dataloader)
    images, labels = next(dataiter)

    images_d1 = images[labels == digit1]
    images_d2 = images[labels == digit2]
    labels_d1 = labels[labels == digit1]
    labels_d2 = labels[labels == digit2]

    half = num_images // 2
    selected_images = torch.cat([images_d1[:half], images_d2[:half]], dim=0).to(device)
    selected_labels = torch.cat([labels_d1[:half], labels_d2[:half]], dim=0)
    
    with torch.no_grad():
        reconstructed, _ = model(selected_images)
    
    selected_images = selected_images.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()
    
    fig, axes = plt.subplots(2, num_images, figsize=(12, 4))
    for i in range(num_images):
        axes[0, i].imshow(selected_images[i].squeeze(), cmap='gray')
        axes[0, i].set_title(f"Original: {selected_labels[i].item()}")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(reconstructed[i].squeeze(), cmap='gray')
        axes[1, i].set_title("Reconstructed")
        axes[1, i].axis('off')
        
    plt.tight_layout()
    plt.savefig("mnist_reconstruction.png")
    print("--> Saved balanced reconstruction plot to 'mnist_reconstruction.png'")
    plt.show()

plot_reconstruction(model, train_loader, digit1=0, digit2=4)

torch.save(model.state_dict(), "ae_swd.pt")
print("--> Saved Phase 2 Autoencoder weights to 'ae_swd.pt'")