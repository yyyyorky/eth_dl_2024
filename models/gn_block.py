#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import functools
import torch
from torch import nn
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Size

import sys
import os
util_path = '../utils/'
sys.path.append(os.path.abspath(util_path))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class GraphNetBlock(MessagePassing):
    '''
    '''
    def __init__(self, node_processor_fn, edge_processor_fn):
        super().__init__(aggr = 'add')
        self.node_processor = node_processor_fn()
        self.mesh_edge_processor = edge_processor_fn()
        self.world_edge_processor = edge_processor_fn()

        self.inspector.inspect(self.message_mesh)
        self.inspector.inspect(self.message_world_direct)

        self.__user_args__ = self.inspector.keys(
            ['message_mesh', 'message_world_direct', 'aggregate', 'update']).difference(
            self.special_args)

    def forward(self, sample: HeteroData):
        sample = self.propagate(sample)
        return sample

    def message_mesh(self, node_features_i, node_features_j, edge_features):
        '''
        message from fluid to fluid
        '''
        in_features = torch.cat([node_features_i, node_features_j, edge_features], dim=-1)
        out_features = self.mesh_edge_processor(in_features)
        return out_features
    
    def message_world_direct(self, fluid_features_i, env_features_j, edge_features):
        '''
        message from environment to fluid
        '''
        in_features = torch.cat([fluid_features_i, env_features_j, edge_features], dim=-1)
        out_features = self.world_edge_processor(in_features)
        return out_features

    
    def update_mesh_edge_features(self, sample: HeteroData):
        node_features = sample['fluid'].node_attr
        edge_features = sample['fluid', 'm_e', 'fluid'].edge_attr
        edge_index = sample['fluid', 'm_e', 'fluid'].edge_index
        size = self._check_input(edge_index, None)
        coll_dict = self._collect(self.__user_args__, edge_index,
                                     size, dict(node_features=node_features))
        coll_dict['edge_features'] = edge_features
        msg_kwargs = self.inspector.distribute('message_mesh', coll_dict)
        out = self.message_mesh(**msg_kwargs)
        return out
    
    def update_world_edge_direct_features(self, sample):
        edge_index = sample['env', 'wm_e', 'fluid'].edge_index
        edge_features = sample['env', 'wm_e', 'fluid'].edge_attr
        fluid_features = sample['fluid'].node_attr
        env_features = sample['env'].node_attr
        N_fluid = fluid_features.shape[0]
        N_env = env_features.shape[0]
        size = (N_env, N_fluid)

        size = self._check_input(edge_index, size)
        __user_args__ = self.inspector.keys(['message_world_direct']).difference(self.special_args)
        coll_dict = self._collect(__user_args__, edge_index,
                                     size, dict(fluid_features=fluid_features, env_features=env_features))
        coll_dict['edge_features'] = edge_features
        msg_kwargs = self.inspector.distribute('message_world_direct', coll_dict)
        out = self.message_world_direct(**msg_kwargs)
        return out
    
    def aggregate_nodes(self, edge_features, edge_index, user_args, size, **kwargs):
        coll_dict = self._collect(user_args, edge_index,
                                     size, kwargs)
        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        node_features = self.aggregate(edge_features, **aggr_kwargs)
        return node_features
    
    def update(self, aggregated_features_mesh, aggregated_features_world, features):
        input_features = torch.cat([aggregated_features_mesh, aggregated_features_world, features], dim=1)
        out_features = self.node_processor(input_features)
        return out_features
    
    def propagate(self, sample: HeteroData):
        N_fluid = sample['fluid'].node_attr.shape[0]
        N_env = sample['env'].node_attr.shape[0]

        # Update mesh edge features
        mesh_edge_features_updated = self.update_mesh_edge_features(sample)
        world_direct_features_updated = self.update_world_edge_direct_features(sample)

        aggr_args = self.inspector.keys(['aggregate']).difference(self.special_args)
        mesh_edge_index = sample['fluid', 'm_e', 'fluid'].edge_index
        mesh_size = (N_fluid, N_fluid)
        fluid_features_from_mesh = self.aggregate_nodes(mesh_edge_features_updated, mesh_edge_index, aggr_args,
                                                        mesh_size)
        
        world_direct_edge_index = sample['env', 'wm_e', 'fluid'].edge_index
        world_direct_size = (N_env, N_fluid)
        fluid_features_from_world = self.aggregate_nodes(world_direct_features_updated, world_direct_edge_index,
                                                         aggr_args, world_direct_size)
        
        fluid_features = sample['fluid'].node_attr
        env_features = sample['env'].node_attr

        fluid_features_new = self.update(fluid_features_from_mesh, fluid_features_from_world, fluid_features)

        fluid_features_new = fluid_features + fluid_features_new

        sample['fluid'].node_attr = fluid_features_new

        mesh_edge_features_new = mesh_edge_features_updated + sample['fluid', 'm_e', 'fluid'].edge_attr
        world_direct_features_new = world_direct_features_updated + sample['env', 'wm_e', 'fluid'].edge_attr

        sample['fluid', 'm_e', 'fluid'].edge_attr = mesh_edge_features_new
        sample['env', 'wm_e', 'fluid'].edge_attr = world_direct_features_new

        return sample


         
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.dataset import EncoderDecoderDataset

    dataset = EncoderDecoderDataset()
    sample = dataset[0]
    model = GraphNetBlock(node_processor_fn=functools.partial(nn.Linear, in_features=9, out_features=3),
                            edge_processor_fn=functools.partial(nn.Linear, in_features=9, out_features=3)).to('cuda')
    out = model(sample)


# %%
