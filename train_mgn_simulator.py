#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from models.mgn import MeshGraphNet
import torch
import torch.nn as nn
from utils.dataset import OneStepGraphDataset, TemporalSequenceGraphDataset
from torch_geometric.loader import DataLoader
from utils.constant import Constant
from utils.rollout import rollout
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

C = Constant()
retrain = False

def set_seed(seed = C.seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

set_seed()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
train_dataset = OneStepGraphDataset(split='train')
train_loader = DataLoader(train_dataset, batch_size=C.batch_size, shuffle=True)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Initialize the model
model = MeshGraphNet(   output_size = C.output_size,
                        latent_size = C.latent_size,
                        num_layers = C.num_layers,
                        n_nodefeatures = C.node_features,
                        n_edgefeatures_mesh = C.edge_features,
                        n_edgefeatures_world = C.edge_features,
                        message_passing_steps = C.message_passing_steps
                        ).to(C.device)

optimizer = torch.optim.Adam(model.parameters(), lr=C.lr)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999999)


#training loop
state_dict = None
if retrain:
    model.train()
    for epoch in range(C.num_epochs):
        print(f'Epoch {epoch} out of {C.num_epochs}')
        epoch_loss = 0
        for i, sample in enumerate(tqdm(train_loader)):
            sample = sample.to(C.device)
            optimizer.zero_grad()
            out = model(sample)
            loss = criterion(out['fluid'].node_attr, sample['fluid'].node_target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch loss: {epoch_loss}')
        scheduler.step()
else:
    model.load_state_dict(torch.load(C.data_dir + 'checkpoints/mgn_sim.pth', weights_only=True))

model.eval()
state_dict = model.state_dict()
save_path = C.data_dir + 'checkpoints/mgn_sim.pth'
torch.save(state_dict, save_path)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#one step prediction
test_dataset = OneStepGraphDataset(split='test')
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
relative_l2_error = 0
model.eval()
with torch.no_grad():
    for i, sample in enumerate(tqdm(test_loader)):
        out = model(sample)
        denominator = torch.norm(sample['fluid'].node_target, dim=1)
        numerator = torch.norm(out['fluid'].node_attr - sample['fluid'].node_target, dim=1)
        relative_l2_error += torch.mean(numerator / denominator).item()

print(f'Relative L2 error: {relative_l2_error / len(test_loader)}')


# %%
#rollout prediction
test_dataset = TemporalSequenceGraphDataset(split='test')
num_steps = test_dataset[0]['fluid'].node_target.shape[0]
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
relative_l2_error_hist = torch.zeros(num_steps, dtype=torch.float32, device=C.device)
model.eval()
with torch.no_grad():
    for i, sample in enumerate(tqdm(test_loader)):
        result = rollout(model, sample, num_steps)
        target = sample['fluid'].node_target
        denominator = torch.norm(target, dim=1)
        numerator = torch.norm(result - target, dim=1)
        relative_l2_error_hist += torch.mean(numerator / denominator, dim=-1)
relative_l2_error_hist /= len(test_loader)
#%%
plt.plot(relative_l2_error_hist.cpu().numpy())




# %%
