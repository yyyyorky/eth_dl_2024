#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData, Batch
from typing import Union

from utils.constant import Constant
from copy import deepcopy

C = Constant()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
@torch.no_grad()
def rollout(model: nn.Module,
            initial: Batch,
            num_steps: int,
            device: str = C.device
            ) -> Batch:
    """
    in rollout we predict the future states of the system given the initial state
    we use target to store the predicted states for convenience
    """
    result_list = []
    sample = initial
    for i in range(num_steps):
        out = model(sample)
        result_list.append(out['fluid'].node_attr)
        sample['fluid'].node_attr = out['fluid'].node_attr
    result = torch.stack(result_list, dim=0)
    return result

    
    

