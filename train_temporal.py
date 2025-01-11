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
# Initialize the model
position_mesh = torch.from_numpy(np.loadtxt(os.path.join(C.data_dir, "meshPosition_all.txt"))).to(C.device)
position_pivotal = torch.from_numpy(np.loadtxt(os.path.join(C.data_dir, "meshPosition_pivotal.txt"))).to(C.device)
tsl_dataset = TemporalSequenceLatentDataset( 
                                            split='train', 
                                            position_mesh=position_mesh, 
                                            position_pivotal=position_pivotal,
                                            produce_latent=False)

tsl_loader = DataLoader(tsl_dataset, batch_size=len(tsl_dataset), shuffle=True)

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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.train()
epochs = C.num_epochs*1000
print_freq = 20
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=C.lr*0.1,        # Lower learning rate
    weight_decay=0.1,   # Increase regularization
    betas=(0.9, 0.98)  # Adjust momentum
)
criterion = nn.MSELoss()
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=100,                # Shorter cycles for transformers
    eta_min=1e-5,
    T_mult=2
)
scaler = GradScaler()

for epoch in tqdm(range(epochs)):
   epoch_loss = 0
   for x, context in tsl_loader:
       input = x[:, :C.time_steps-1]
       target = x[:, 1:]
       optimizer.zero_grad()
       
       # Forward pass with autocast for mixed precision
       with autocast():
           out = model(input, context)
           loss = criterion(out, target)
       
       # Scale loss and compute gradients
       scaler.scale(loss).backward()
       
       # Unscale gradients and clip
       scaler.unscale_(optimizer)
       torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
       
       # Update weights
       scaler.step(optimizer)
       scaler.update()
       
       epoch_loss += loss.detach().item()
   
   scheduler.step()
   if epoch % print_freq == 0:
       print(f'Epoch {epoch}/{epochs}, loss: {epoch_loss:.6f}, lr: {scheduler.get_last_lr()[0]:.6f}')

torch.save(model.state_dict(), os.path.join(C.data_dir, 'checkpoints', 'sequence_model_spatial.pth'))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


