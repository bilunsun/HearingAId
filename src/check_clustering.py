import torch
import matplotlib.pyplot as plt
import numpy as np
import umap
from tqdm.auto import tqdm

from lib import AudioDataModule
from train import Model

datamodule = AudioDataModule("custom", batch_size=32, shuffle=False, num_workers=0, target_sample_rate=16_000, n_samples=64_000)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model.load_from_checkpoint("checkpoints/good-butterfly-178.ckpt")
model = model.to(device)
model.scaler.to(device)
model = model.eval()

classes = [datamodule.train_dataset.dataset.CLASS_ID_TO_NAME[i] for i in range(len(datamodule.train_dataset.dataset.CLASS_ID_TO_NAME))]
# assert len(classes) == model.hparams.n_classes

@torch.no_grad()
def get_embeddings_and_labels():
    embeddings = []
    labels = []
    for x, y in tqdm(datamodule.train_dataloader()):
        x = x.to(device)
        x = model.scaler.transform(x)
        pred = model.model.backbone(x)
        pred = pred.reshape(pred.size(0), -1)
        # pred = model.model.classifier[:-2](pred)
        # pred = model(x)
        embeddings.append(pred)
        labels.append(y)

    embeddings = torch.cat(embeddings)
    labels = torch.cat(labels)

    embeddings = embeddings.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    return embeddings, labels


embeddings, labels = get_embeddings_and_labels()
class_ids = [classes[l] for l in labels]

reducer = umap.UMAP()
projections = reducer.fit_transform(embeddings)
x = projections[:, 0]
y = projections[:, 1]

for i, c in enumerate(classes):
    mask = np.where(labels == i)
    plt.scatter(x[mask], y[mask], label=c)
plt.legend()
plt.show()
