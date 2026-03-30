import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset_mnist import get_mnist_subset
from model_mnist import MNISTAutoencoder, LatentGenerator


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


train_loader = get_mnist_subset(digit1=0, digit2=4, batch_size=64)


ae_model = MNISTAutoencoder(latent_dim=16).to(device)

ae_model.load_state_dict(torch.load("ae_swd.pt"))
ae_model.eval() 


generator = LatentGenerator(noise_dim=16, latent_dim=16).to(device)
optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3)

EPOCHS = 10

# -------- TRAINING LOOP --------
print("Starting Generator Training...")
for epoch in range(EPOCHS):
    generator.train()
    total_swd_loss = 0
    
    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.to(device)
        batch_size = images.size(0)
        
        
        with torch.no_grad():
            _, real_latent = ae_model(images)
            
        
        raw_noise = torch.randn(batch_size, 16, device=device)
        generated_latent = generator(raw_noise)
        
        
        loss = sliced_wasserstein_distance(generated_latent, real_latent, device=device)
        
      
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_swd_loss += loss.item()
        
    avg_loss = total_swd_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Generator SWD Loss: {avg_loss:.4f}")

# -------- VISUALIZATION: 
def generate_new_digits(generator, decoder, num_images=10):
    generator.eval()
    decoder.eval() 
    
    
    noise = torch.randn(num_images, 16, device=device)
    
    with torch.no_grad():
       
        mapped_latent = generator(noise)
        
        hallucinated_images = decoder(mapped_latent)
        
    hallucinated_images = hallucinated_images.cpu().numpy()
    
    # Plot the entirely newly created digits
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i in range(num_images):
        axes[i].imshow(hallucinated_images[i].squeeze(), cmap='gray')
        axes[i].axis('off')
        
    plt.suptitle("Pure Hallucinations: Noise -> Generator -> Decoder")
    plt.tight_layout()
    plt.savefig("mnist_generated_bonus.png")
    print("--> Saved generated digits to 'mnist_generated_bonus.png'")
    plt.show()

# Execute the final generation step
generate_new_digits(generator, ae_model.decoder)