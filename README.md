# ETH DL 2024 project: Spatial and Temporal Attention to Improve Graph-Based Physical Simulation

If you have any questions or problems regarding reproduction, please contact <yiyyan@ethz.ch>.

## Data

Download data from https://polybox.ethz.ch/index.php/s/yZGBaAnA1EfFy1D

Put all the file inside the archive at `./data` so that it looks like

```bash
project_root/
├── data/
│   ├── checkpoints/
│   ├── result/
│   ├── meshgrid/
│   ├── rawData.npy
│   ├── meshPosition_all.txt
│   ├── meshPosition_pivotal.txt
│   └── ...
└── ...
```

Modify `Constant.root_dir` in `utils/constant.py` to the absolute path of project directory.

## Env

Only Linux or WSL2 are supported and tested.

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

## Result GIF animation

The result GIF animation can be found in `./data/meshgrid`. 