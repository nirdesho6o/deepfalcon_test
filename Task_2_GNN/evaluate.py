import torch
from torch_geometric.loader import DataLoader
from dataset_graph import JetGraphDataset
from gnn_model import JetGNN
from sklearn.metrics import roc_auc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = JetGraphDataset("../data/quark-gluon.hdf5", limit=5000)

loader = DataLoader(dataset, batch_size=32)

model = JetGNN().to(device)

model.load_state_dict(torch.load("gnn_model.pt"))

model.eval()

preds = []
labels = []

with torch.no_grad():

    for batch in loader:

        batch = batch.to(device)

        out = model(batch.x, batch.edge_index, batch.batch)

        prob = torch.softmax(out, dim=1)[:,1]

        preds.extend(prob.cpu().numpy())
        labels.extend(batch.y.cpu().numpy())


auc = roc_auc_score(labels, preds)

print("ROC AUC:", auc)