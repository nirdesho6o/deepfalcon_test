import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset_hep import JetImageDataset
from model_hep import HEPAutoencoder, HEPLatentGenerator

def sliced_wasserstein_distance(z, target_z, num_projections=50, device='cpu'):
    latent_dim = z.size(1)
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


ae_model = HEPAutoencoder(latent_dim=128).to(device)
ae_model.load_state_dict(torch.load("hep_ae_swd.pt"))
ae_model.eval()

generator = HEPLatentGenerator(noise_dim=64, latent_dim=128).to(device)
optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3)

EPOCHS = 10


print("Training HEP Latent Generator...")
for epoch in range(EPOCHS):
    generator.train()
    total_swd = 0
    
    for images, _ in train_loader:
        images = images.to(device)
        batch_size = images.size(0)
        
        
        with torch.no_grad():
            _, real_latent = ae_model(images)
            
        
        raw_noise = torch.randn(batch_size, 64, device=device)
        generated_latent = generator(raw_noise)
        
      
        loss = sliced_wasserstein_distance(generated_latent, real_latent, device=device)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_swd += loss.item()
        
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Generator SWD Loss: {total_swd/len(train_loader):.4f}")

def plot_hallucinated_jets(generator, decoder, num_images=4):
    generator.eval()
    
    
    noise = torch.randn(num_images, 64, device=device)
    
    with torch.no_grad():
        mapped_latent = generator(noise)
        hallucinated_images = decoder(mapped_latent)
        
    hallucinated_images = hallucinated_images.cpu().numpy()
    
    def normalize_for_plot(img_array):
        import numpy as np
        img_array = np.clip(img_array, 0, None)
        vmax = img_array.max(axis=(1,2,3), keepdims=True)
        return img_array / (vmax + 1e-8)

    hallucinated_images = normalize_for_plot(hallucinated_images)

    fig, axes = plt.subplots(1, num_images, figsize=(15, 4))
    for i in range(num_images):
        img = hallucinated_images[i].transpose(1, 2, 0)
        axes[i].imshow(img)
        axes[i].set_title("Hallucinated Jet")
        axes[i].axis('off')
        
    plt.suptitle("Optimal Transport Generation: Noise -> Generator -> Decoder")
    plt.tight_layout()
    plt.savefig("hep_generated_bonus.png")
    print("--> Saved generated jets to 'hep_generated_bonus.png'")
    plt.show()

plot_hallucinated_jets(generator, ae_model.decode)