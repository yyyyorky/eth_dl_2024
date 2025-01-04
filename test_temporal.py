#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import os
import sys
import torch
import torch.nn as nn
from utils.dataset import TemporalSequenceLatentDataset
from torch_geometric.loader import DataLoader
from utils.constant import Constant
import numpy as np
import random
from tqdm import tqdm
from models.sequence_model import SequenceModel
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

C = Constant()

def set_seed(seed = C.seed+6):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

set_seed()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
position_mesh = torch.from_numpy(np.loadtxt(os.path.join(C.data_dir, "meshPosition_all.txt"))).to(C.device)
position_pivotal = torch.from_numpy(np.loadtxt(os.path.join(C.data_dir, "meshPosition_pivotal.txt"))).to(C.device)
tsl_dataset = TemporalSequenceLatentDataset( 
                                            split='test', 
                                            position_mesh=position_mesh, 
                                            position_pivotal=position_pivotal,
                                            produce_latent=False)

tsl_loader = DataLoader(tsl_dataset, batch_size=1, shuffle=True)

model = SequenceModel(
    input_dim=C.token_size,
    input_context_dim= C.context_dim,
    num_layers_decoder=C.temporal_docoder_layers,
    num_heads=8,
    dim_feedforward_scale=8,
    num_layers_context_encoder=C.num_layers,
    num_layers_input_encoder=C.num_layers,
    num_layers_output_encoder=C.num_layers,
).to(C.device)

state_dic = torch.load(os.path.join(C.data_dir, 'checkpoints', 'sequence_model.pth'), weights_only=True)
model.load_state_dict(state_dic)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.eval()
relative_l2_error = 0
with torch.no_grad():
    for i, (z, re) in enumerate(tqdm(tsl_loader)):
        input = z[:, :-1] #shape: [batch_size, seq_len, nodes_num, nodes_features]
        target = z[:, 1:]
        out = model(input, re)
        out = input.reshape(1, -1, C.token_size)
        target = target.reshape(1, -1, C.token_size)
        denominator = torch.norm(target, dim=2)
        numerator = torch.norm(out - target, dim=2)
        error = torch.mean(numerator/denominator).item()
        relative_l2_error += error
    relative_l2_error /= len(tsl_loader)
    print(f'Relative L2 error: {relative_l2_error}')
# %%
