import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset_mnist import get_mnist_subset
from model_mnist import MNISTAutoencoder

# -------- SETUP --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# Load data (Choosing digits 0 and 4 as suggested by the task)
print("Loading MNIST subset...")
train_loader = get_mnist_subset(digit1=0, digit2=4, batch_size=64)

# Initialize model, loss, and optimizer
model = MNISTAutoencoder(latent_dim=16).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 10

# -------- TRAINING LOOP --------
print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for batch_idx, (images, _) in enumerate(train_loader):
        # We don't need labels for an autoencoder, just the images
        images = images.to(device)
        
        # Forward pass
        reconstructed, latent_vector = model(images)
        
        # Compute pixel-wise reconstruction loss
        loss = criterion(reconstructed, images)
        
        # Backward pass & optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] | MSE Loss: {avg_loss:.4f}")

# -------- VISUALIZATION --------
# The mentors explicitly requested a side-by-side comparison
def plot_reconstruction(model, dataloader, digit1=0, digit2=4, num_images=6):
    model.eval()
    
    # Grab one batch of images
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    
    # Separate the batch by digit to ensure we get a mix
    images_d1 = images[labels == digit1]
    images_d2 = images[labels == digit2]
    labels_d1 = labels[labels == digit1]
    labels_d2 = labels[labels == digit2]
    
    # Take half from the first digit, half from the second
    half = num_images // 2
    selected_images = torch.cat([images_d1[:half], images_d2[:half]], dim=0).to(device)
    selected_labels = torch.cat([labels_d1[:half], labels_d2[:half]], dim=0)
    
    with torch.no_grad():
        reconstructed, _ = model(selected_images)
    
    # Move to CPU for matplotlib
    selected_images = selected_images.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()
    
    fig, axes = plt.subplots(2, num_images, figsize=(12, 4))
    for i in range(num_images):
        # Top row: Originals
        axes[0, i].imshow(selected_images[i].squeeze(), cmap='gray')
        axes[0, i].set_title(f"Original: {selected_labels[i].item()}")
        axes[0, i].axis('off')
        
        # Bottom row: Reconstructions
        axes[1, i].imshow(reconstructed[i].squeeze(), cmap='gray')
        axes[1, i].set_title("Reconstructed")
        axes[1, i].axis('off')
        
    plt.tight_layout()
    plt.savefig("mnist_reconstruction_swd.png")
    print("--> Saved balanced reconstruction plot to 'mnist_reconstruction_swd.png'")
    plt.show()

# Run the visualization after training
plot_reconstruction(model, train_loader, digit1=0, digit2=4)