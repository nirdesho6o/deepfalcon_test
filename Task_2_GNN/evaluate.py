import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score

from dataset_graph import JetGraphDataset
from gnn_model import JetGNN

device = torch.device("cpu")


torch.manual_seed(42)


LIMIT = 5000
SPLIT = int(LIMIT * 0.8)

dataset = JetGraphDataset("../data/quark-gluon.hdf5", limit=LIMIT)


dataset = dataset.shuffle() 

test_dataset = dataset[SPLIT:]
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# -------- MODEL --------
model = JetGNN().to(device)

model.load_state_dict(torch.load("gnn_model.pt"))
model.eval()

# -------- EVALUATION --------
preds = []
labels = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)

        out = model(batch.x, batch.edge_index, batch.batch)
        prob = torch.softmax(out, dim=1)[:, 1]

        preds.extend(prob.cpu().numpy())
        labels.extend(batch.y.cpu().numpy())

auc = roc_auc_score(labels, preds)
print(f"Final Test ROC AUC: {auc:.4f}")