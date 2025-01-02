# eth_dl_2024

# Data 

Download data from https://polybox.ethz.ch/index.php/s/cnHDWlZ9uePKhso

Put the data at `./data` so that it looks like

project_root/

├── data/

│   ├── rawData.npy

│   └── ...

└── ...

# Env

conda env create -f eth_dl.yml

conda activate eth_dl

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html