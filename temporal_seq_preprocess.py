#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import os
import sys
from models.autoencoder_skip_connection import MeshReduce
import torch
import torch.nn as nn
from utils.dataset import TemporalSequenceLatentDataset, EncoderDecoderDataset
from torch_geometric.loader import DataLoader
from utils.constant import Constant
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

C = Constant()

def set_seed(seed = C.seed+11):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

set_seed()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Initialize the model
position_mesh = torch.from_numpy(np.loadtxt(os.path.join(C.data_dir, "meshPosition_all.txt"))).to(C.device)
position_pivotal = torch.from_numpy(np.loadtxt(os.path.join(C.data_dir, "meshPosition_pivotal.txt"))).to(C.device)

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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tsl_dataset = TemporalSequenceLatentDataset(encoder=model, 
                                            split='train', 
                                            position_mesh=position_mesh, 
                                            position_pivotal=position_pivotal,
                                            produce_latent=True)

tsl_dataset = TemporalSequenceLatentDataset(encoder=model, 
                                            split='test', 
                                            position_mesh=position_mesh, 
                                            position_pivotal=position_pivotal,
                                            produce_latent=True)
print(tsl_dataset[0][0].shape)
# %%
