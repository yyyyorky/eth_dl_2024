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
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rollout_error = 0
with torch.no_grad():
    for i, (z, re) in enumerate(tqdm(tsl_loader)):
        input = z[:, 0].unsqueeze(1)  # shape: [batch_size, seq_len, nodes_num, nodes_features]
        target = z[:, 1:]  # true future sequence
        out = input  # start with the given initial input

        # Rollout: Iteratively predict the future sequence
        predicted_sequence = []
        for t in range(target.shape[1]):  # iterate over the target sequence length
            next_out = model(out, re)  # generate one step ahead
            predicted_sequence.append(next_out[:, -1:].clone())  # append the last predicted step
            out = torch.cat((out, next_out[:, -1:]), dim=1)  # update input for the next step

        # Stack the predicted sequence into the same shape as the target
        predicted_sequence = torch.cat(predicted_sequence, dim=1)  # shape: [batch_size, seq_len, ...]

        # Reshape for error calculation
        predicted_sequence = predicted_sequence.reshape(1, -1, C.token_size)
        target = target.reshape(1, -1, C.token_size)

        # Calculate rollout error
        denominator = torch.norm(target, dim=2) + 1e-8  # add epsilon to avoid division by zero
        numerator = torch.norm(predicted_sequence - target, dim=2)
        error = torch.mean(numerator / denominator).item()
        rollout_error += error

    rollout_error /= len(tsl_loader)
    print(f'Rollout Error: {rollout_error}')
