# eth_dl_2024

# Data 

Download data from https://polybox.ethz.ch/index.php/s/yZGBaAnA1EfFy1D

Put all the file inside the archive at `./data` so that it looks like

```bash
project_root/
├── data/
│   ├── rawData.npy
│   └── ...
└── ...
```

Before run model, please compute `latent_test.pt` and `latent_train.pt` using `utils.dataset.TemporalSequenceLatentDataset` with `produce_latent = True`

# Env
Create environment using conda

```bash
conda env create -f eth_dl.yml

conda activate eth_dl
```
Then install `torch_cluster` and `torch_scatter`. It is recommended to use conda.

```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```
or

```bash
conda install pytorch-scatter -c pyg
conda install pytorch-cluster -c pyg
```
