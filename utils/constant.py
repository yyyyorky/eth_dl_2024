import torch

class Constant:
    def __init__(self):
        self.root_dir = '/home/york_ubuntu_wsl/deepl_project/eth_dl_2024/'
        self.seed = 42
        self.data_dir = self.root_dir + '/data/'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = 32
        self.node_features = 3
        self.edge_features = 3
        self.message_passing_steps = 16
        self.num_layers = 2
        self.latent_size = 64
        self.output_size = 3
        self.lr = 1e-3
        self.num_epochs = 200
        self.nodes_per_mesh = 1699


if __name__ == "__main__":
    pass