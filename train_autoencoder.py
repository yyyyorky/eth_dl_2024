#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import os
import sys
from models.autoencoder import MeshReduce
import torch
import torch.nn as nn
from utils.dataset import EncoderDecoderDataset
from torch_geometric.loader import DataLoader
from utils.constant import Constant
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

C = Constant()

def set_seed(seed = C.seed+7):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

set_seed()

retrain = True
batch_size = C.batch_size
num_epochs = C.num_epochs * 2

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
train_dataset = EncoderDecoderDataset()
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

position_mesh = torch.from_numpy(np.loadtxt(os.path.join(C.data_dir, "meshPosition_all.txt"))).to(C.device)
position_pivotal = torch.from_numpy(np.loadtxt(os.path.join(C.data_dir, "meshPosition_pivotal.txt"))).to(C.device)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Initialize the model
model = MeshReduce(
    input_node_features_dim=C.node_features,
    input_edge_features_dim=C.edge_features,
    output_node_features_dim=C.node_features,
    internal_width=C.latent_size,
    message_passing_steps=C.message_passing_steps,
    num_layers=C.num_layers
).to(C.device)

optimizer = torch.optim.Adam(model.parameters(), lr=C.lr)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999999)


if retrain:
    model.train()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch} out of {num_epochs}')
        epoch_loss = 0
        for i, sample in enumerate(tqdm(train_loader)):
            sub_batch_size = sample['fluid'].node_attr.shape[0] // C.nodes_per_mesh
            sample = sample.to(C.device)
            optimizer.zero_grad()
            out = model(sample, position_mesh, position_pivotal, sub_batch_size)
            loss = criterion(out['fluid'].node_attr, sample['fluid'].node_target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch loss: {epoch_loss}')
        scheduler.step()
else:
    model.load_state_dict(torch.load(C.data_dir + 'checkpoints/autoencoder.pth', weights_only=True))

model.eval()
state_dict = model.state_dict()
save_path = C.data_dir + 'checkpoints/autoencoder.pth'
torch.save(state_dict, save_path)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%