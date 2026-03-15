import matplotlib.pyplot as plt
from dataset import JetDataset

dataset = JetDataset("../data/quark-gluon.hdf5")

x = dataset[0].numpy()

fig, axs = plt.subplots(1,3, figsize=(9,3))

titles = ["ECAL","HCAL","Tracks"]

for i in range(3):
    axs[i].imshow(x[i])
    axs[i].set_title(titles[i])
    axs[i].axis("off")

plt.tight_layout()
plt.savefig("figures/sample_jet_event.png", dpi=300)
plt.show()