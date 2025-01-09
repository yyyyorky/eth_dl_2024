#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import os
import sys
import torch
import torch.nn as nn
from utils.dataset import TemporalSequenceLatentDataset, EncoderDecoderDataset
from torch_geometric.loader import DataLoader
from utils.constant import Constant
import numpy as np
import random
from tqdm import tqdm
from models.sequence_model import SequenceModel
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from models.autoencoder import MeshReduce
from matplotlib import pyplot as plt

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
test_dataset = EncoderDecoderDataset(split='test')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
seq_model = SequenceModel(
    input_dim=C.token_size,
    input_context_dim= C.context_dim,
    num_layers_decoder=C.temporal_docoder_layers*2,
    num_heads=8,
    dim_feedforward_scale=8,
    num_layers_context_encoder=C.num_layers,
    num_layers_input_encoder=C.num_layers,
    num_layers_output_encoder=C.num_layers,
).to(C.device)

state_dic = torch.load(os.path.join(C.data_dir, 'checkpoints', 'sequence_model_backup.pth'), weights_only=True)
seq_model.load_state_dict(state_dic)
seq_model.eval()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mesh_reduced_model = MeshReduce(
    input_node_features_dim=C.node_features,
    input_edge_features_dim=C.edge_features,
    output_node_features_dim=C.node_features,
    internal_width=C.latent_size,
    message_passing_steps=C.message_passing_steps*2,
    num_layers=C.num_layers
).to(C.device)

state_dic = torch.load(os.path.join(C.data_dir, 'checkpoints', 'autoencoder_backup.pth'), weights_only=True)
mesh_reduced_model.load_state_dict(state_dic)
mesh_reduced_model.eval()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ReAll = torch.from_numpy(np.linspace(300, 1000, 101)).float().reshape([-1, 1]).to(C.device)
ReAll = ReAll / ReAll.max()
index = [i for i in range(101) if i % 2 == 1]
ReAll = ReAll[index]
sequence_len = 401

target_sequence = []

def denormalize(invar, std, mu):
    denormalized_invar = invar * std + mu
    return denormalized_invar

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Initialize rollout errors for different quantities (u, v, p)
rollout_error_u = 0
rollout_error_v = 0
rollout_error_p = 0

sum_rollout_error_u = 0
sum_rollout_error_v = 0
sum_rollout_error_p = 0
iteration = 0
error_record = torch.zeros(sequence_len-1, len(test_loader)//sequence_len, 3)

# Disable gradient computation for efficiency
with torch.no_grad():
    # Iterate over the test dataset
    for i, sample in enumerate(tqdm(test_loader)):
        # Encode the current sample using the mesh-reduced model
        z_sample = mesh_reduced_model.encode(sample, position_mesh, position_pivotal, 1).clone()

        sid = i // sequence_len
        tid = i % sequence_len
        
        # Check if we are at the start of a new sequence
        if i % sequence_len == 0:
            # Use the initial input (z_0) as the starting point (t = 0)
            z_0 = z_sample['fluid'].node_attr
            out = z_0.unsqueeze(0).unsqueeze(0)

            # Rollout: Predict the future sequence iteratively
            for t in range(sequence_len):  # Predict for the entire target sequence length
                next_out = seq_model(out, ReAll[i // sequence_len].unsqueeze(0))  # Generate next step
                # latent state (t = 0, 1, 2, 3, 4, ...) at the same Reynolds number
                out = torch.cat((out, next_out[:, -1:]), dim=1)  # Update the input with the new prediction
        # (t = 1, 2, 3, 4, ...) at the same Reynolds number
        else:
            # Prepare the predicted latent state (t = 1, 2, 3, 4, ...) at the same Reynolds number
            z_pred = z_sample.clone()
            z_pred['fluid'].node_attr = out[:, i % sequence_len].squeeze()
            
            # Decode the predicted latent state back to the physical domain
            decode_out = mesh_reduced_model.decode(z_pred, position_mesh, position_pivotal, 1).clone()
            
            # Calculate rollout errors for each component (u, v, p)
            
            # Error for u
            denominator_u = torch.norm(denormalize(sample['fluid'].node_attr[:, 0], test_dataset.node_stats['node_std'][0], test_dataset.node_stats['node_mean'][0]))
            numerator_u = torch.norm(denormalize(sample['fluid'].node_attr[:, 0], test_dataset.node_stats['node_std'][0], test_dataset.node_stats['node_mean'][0]) - 
                                       denormalize(decode_out['fluid'].node_attr[:, 0], test_dataset.node_stats['node_std'][0], test_dataset.node_stats['node_mean'][0]))
            error_u = torch.mean(numerator_u / denominator_u).item()
            error_record[tid-1, sid, 0] = error_u
            rollout_error_u += error_u      
            
            # Error for v
            denominator_v = torch.norm(denormalize(sample['fluid'].node_attr[:, 1], test_dataset.node_stats['node_std'][1], test_dataset.node_stats['node_mean'][1]))
            numerator_v = torch.norm(denormalize(sample['fluid'].node_attr[:, 1], test_dataset.node_stats['node_std'][1], test_dataset.node_stats['node_mean'][1]) - 
                                       denormalize(decode_out['fluid'].node_attr[:, 1], test_dataset.node_stats['node_std'][1], test_dataset.node_stats['node_mean'][1]))
            error_v = torch.mean(numerator_v / denominator_v).item()
            error_record[tid-1, sid, 1] = error_v
            rollout_error_v += error_v
            
            # Error for p
            denominator_p = torch.norm(denormalize(sample['fluid'].node_attr[:, 2], test_dataset.node_stats['node_std'][2], test_dataset.node_stats['node_mean'][2]))
            numerator_p = torch.norm(denormalize(sample['fluid'].node_attr[:, 2], test_dataset.node_stats['node_std'][2], test_dataset.node_stats['node_mean'][2]) - 
                                       denormalize(decode_out['fluid'].node_attr[:, 2], test_dataset.node_stats['node_std'][2], test_dataset.node_stats['node_mean'][2]))
            error_p = torch.mean(numerator_p / denominator_p).item()
            error_record[tid-1, sid, 2] = error_p
            rollout_error_p += error_p
        
        # At the end of the sequence, normalize and print the accumulated errors
        if i % sequence_len == sequence_len - 1:
            rollout_error_u /= (sequence_len - 1)
            rollout_error_v /= (sequence_len - 1)
            rollout_error_p /= (sequence_len - 1)
            print(f'At {i}, Rollout Error u: {rollout_error_u}, Rollout Error v: {rollout_error_v}, Rollout Error p: {rollout_error_p}')
            
            sum_rollout_error_u += rollout_error_u
            sum_rollout_error_v += rollout_error_v
            sum_rollout_error_p += rollout_error_p
            
            iteration += 1
            
            # Reset errors for the next sequence
            rollout_error_u = 0
            rollout_error_v = 0
            rollout_error_p = 0
    
    print(f'Totally, Rollout Error u: {sum_rollout_error_u / iteration}, Rollout Error v: {sum_rollout_error_v / iteration}, Rollout Error p: {sum_rollout_error_p / iteration}')


# %%
error_hist = error_record.mean(dim=-1).mean(dim=-1).cpu().numpy()
plt.plot(error_hist)
np.save(C.data_dir + 'result/rollout_error_temporal.npy', error_hist)



# %%
