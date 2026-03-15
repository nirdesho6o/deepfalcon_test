import torch
from torch_geometric.loader import DataLoader
from dataset_graph import JetGraphDataset
from gnn_model import JetGNN
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = JetGraphDataset("../data/quark-gluon.hdf5", limit=5000)

loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = JetGNN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 10


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    loop = tqdm(loader)

    for batch in loop:

        batch = batch.to(device)

        out = model(batch.x, batch.edge_index, batch.batch)

        loss = torch.nn.functional.cross_entropy(out, batch.y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        loop.set_description(f"Epoch {epoch}")
        loop.set_postfix(loss=loss.item())

    print("avg loss:", total_loss / len(loader))


torch.save(model.state_dict(), "gnn_model.pt")