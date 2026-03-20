import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from dataset_graph import JetGraphDataset
from gnn_model import JetGNN
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- DATA --------
LIMIT = 5000
SPLIT = int(LIMIT * 0.8)

# Load dataset
dataset = JetGraphDataset("../data/quark-gluon.hdf5", limit=LIMIT)

# CRITICAL FIX: Shuffle indices before splitting to prevent class imbalance
dataset = dataset.shuffle()

train_dataset = dataset[:SPLIT]
val_dataset   = dataset[SPLIT:]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)

# -------- MODEL --------
model = JetGNN().to(device)
# Lowered learning rate for stability with GNNs
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
# Added scheduler to reduce learning rate if validation loss plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

EPOCHS = 15
best_val_auc = 0.0

# -------- TRAINING LOOP --------
for epoch in range(EPOCHS):

    # ---- TRAIN ----
    model.train()
    total_train_loss = 0
    loop = tqdm(train_loader)

    for batch in loop:
        batch = batch.to(device)

        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        loop.set_description(f"Epoch {epoch} [Train]")
        loop.set_postfix(loss=loss.item())

    avg_train_loss = total_train_loss / len(train_loader)

    # ---- VALIDATION ----
    model.eval()
    total_val_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            
            # Track validation loss
            loss = F.cross_entropy(out, batch.y)
            total_val_loss += loss.item()

            # Get probabilities for class 1 (Gluon/Quark depending on your label map) for AUC
            probs = F.softmax(out, dim=1)[:, 1]
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)
    
    # Calculate Metrics
    val_auc = roc_auc_score(all_labels, all_preds)
    
    print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val AUC: {val_auc:.4f}")

    # Step the scheduler based on validation loss
    scheduler.step(avg_val_loss)

    # -------- SAVE BEST MODEL --------
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), "gnn_model.pt")
        print("--> Saved new best model!")

print(f"Training complete. Best Validation AUC: {best_val_auc:.4f}")