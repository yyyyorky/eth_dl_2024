import torch

class Constant:
    def __init__(self):
        self.data_dir = '../data'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    pass