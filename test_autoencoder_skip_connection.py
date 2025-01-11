#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import os
import sys
from models.autoencoder_skip_connection import MeshReduce
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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
test_dataset = EncoderDecoderDataset(split='test')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

position_mesh = torch.from_numpy(np.loadtxt(os.path.join(C.data_dir, "meshPosition_all.txt"))).to(C.device)
position_pivotal = torch.from_numpy(np.loadtxt(os.path.join(C.data_dir, "meshPosition_pivotal.txt"))).to(C.device)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model = MeshReduce(
    input_node_features_dim=C.node_features,
    input_edge_features_dim=C.edge_features,
    output_node_features_dim=C.node_features,
    internal_width=C.latent_size,
    message_passing_steps=C.message_passing_steps,
    num_layers=C.num_layers
).to(C.device)

state_dic = torch.load(os.path.join(C.data_dir, 'checkpoints', 'autoencoder_skip.pth'), weights_only=True)

model.load_state_dict(state_dic)
model.eval()

relative_l2_error = 0
with torch.no_grad():
    for i, sample in enumerate(tqdm(test_loader)):
        out = model(sample, position_mesh, position_pivotal, 1)
        denominator = torch.norm(sample['fluid'].node_attr, dim=1)
        numerator = torch.norm(out['fluid'].node_attr - sample['fluid'].node_target, dim=1)
        error = torch.mean(numerator/denominator).item()
        relative_l2_error += error
    relative_l2_error /= len(test_loader)
    print(f'Relative L2 error: {relative_l2_error}')
# %%
