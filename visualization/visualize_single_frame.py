import argparse
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)
from utils.constant import Constant
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx
import numpy as np
import json
from matplotlib.animation import FuncAnimation
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
def parse_args():
    parser = argparse.ArgumentParser(description="Smooth Mesh Animation")
    parser.add_argument('--param', type=int, help='Param idx: from 0 to 49',default=0)
    parser.add_argument('--result_file_name', type=str,help="mgn result file name(Please put the result file in the data folder)"
    ,default='rollout_result_vanilla_mgn.npy')
    parser.add_argument('--frame_idx', type=int)
    return parser.parse_args()


def main():

    args = parse_args()
    C = Constant()
    param_idx = args.param
    frame_idx = args.frame_idx
    result_file_name = C.data_dir + 'result/'+args.result_file_name
    param_idx_ls = [i for i in range(101) if i % 2 ==1]
    data_path = C.data_dir + 'rawData.npy'
    raw_data = np.load(data_path,allow_pickle=True)
    position_data = np.loadtxt(C.data_dir +'meshPosition_all.txt')
    node_states = raw_data['x'][param_idx_ls][param_idx]
    node_masses = raw_data['mass']
    edge_attrs = raw_data['edge_attr']
    edge_index = raw_data['edge_index']
    node_positions = position_data.copy()


    # FOR test set
    with open(C.data_dir +'node_stats.json','r') as f:
        node_stats = json.load(f)
    node_mean = node_stats['node_mean']
    node_std = node_stats['node_std']

    # test rollout result
    result_path = result_file_name
    if args.result_file_name == 'rollout_result_vanilla_mgn.npy':
        node_states_predict = np.load(result_path, allow_pickle=True)*node_std + node_mean
    else:
        node_states_predict = np.load(result_path, allow_pickle=True)
    node_states_predict = np.concatenate((raw_data['x'][param_idx_ls][:,0,:,:][:,np.newaxis,:,:],node_states_predict ), axis=1)[param_idx]

    # Define figure and axes
    fig, axes = plt.subplots(3, 2, figsize=(24, 18))

    model = result_path.split('_')[-1].split('.')[0]
    if model == 'mgn':
        titles = [
            r'Ground Truth: $u$', r'MeshGraphNet: $u$',
            r'Ground Truth: $v$', r'MeshGraphNet: $v$',
            r'Ground Truth: $p$', r'MeshGraphNet: $p$'
        ]
    else:
        titles = [
            r'Ground Truth: $u$', rf'Temporal-{model}: $u$',
            r'Ground Truth: $v$', rf'Temporal-{model}: $v$',
            r'Ground Truth: $p$', rf'Temporal-{model}: $p$'
        ]


    title_font = {'fontsize': 20, 'fontweight': 'bold', 'family': 'serif'}
    tick_fontsize = 14
    # Customize each subplot
    for i, ax in enumerate(axes.flat):
        ax.set_xlim(-1, 7)
        ax.set_ylim(-2, 2)
        ax.set_title(titles[i]) 

    x_center, y_center = 0,0
    radius = 0.5
    # Create grid for interpolation
    grid_x, grid_y = np.meshgrid(
        
        np.linspace(position_data[:, 0].min()-0.5, position_data[:, 0].max()+0.5, 1000),
        np.linspace(position_data[:, 1].min()-0.5, position_data[:, 1].max()+0.5, 1000)
    )

    # Function to plot the mesh
    def mask_inside_circle(x, y, radius, x_center, y_center):
        distances = np.sqrt((x - x_center)**2 + (y - y_center)**2)
        return distances > radius
    # Function to plot the mesh
    def plot_mesh(ax, position_data, data, title):
        mask = mask_inside_circle(position_data[:, 0], position_data[:, 1], radius, x_center, y_center)
        
        interpolated = griddata(position_data[mask], data[mask], (grid_x, grid_y), method='cubic')

        circle_mask = np.sqrt((grid_x - x_center)**2 + (grid_y - y_center)**2) <= radius
        interpolated[circle_mask] = np.nan

        interpolated = gaussian_filter(interpolated, sigma=0.1)

        # Plot the data
        mesh = ax.pcolormesh(grid_x, grid_y, interpolated, shading='auto', cmap=plt.cm.plasma)
        return mesh

    # Initial meshes
    meshes = []
    for i, ax in enumerate(axes.flat):
        if i % 2 == 0:
            mesh = plot_mesh(ax, position_data, node_states[0, :, i//2], titles[i])
        else:
            mesh = plot_mesh(ax, position_data,node_states_predict[0, :, i//2], titles[i])
            
        for j in range(edge_index.shape[-1]):
            idx = edge_index[:,j]
            x_1, y_1 = position_data[idx[0]]
            x_2, y_2 = position_data[idx[1]]
            ax.plot(
                [x_1, x_2], [y_1, y_2], color='gray', linestyle='--', linewidth=0.5, alpha=0.8
            )
        meshes.append(mesh)



    # To save the animation as a GIF
    plt.savefig(C.data_dir+f'meshgrid/{model}_{frame_idx}.pdf')

    plt.show()

if __name__=="__main__":
    main()
