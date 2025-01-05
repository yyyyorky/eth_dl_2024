import argparse
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.interpolate import griddata

def parse_args():
    parser = argparse.ArgumentParser(description="Smooth Mesh Animation")
    parser.add_argument('--param', type=int, help='Param idx: from 0 to 49',default=0)
    parser.add_argument('--result_file_name', type=str,help="mgn result file name(Please put the result file in the data folder)"
    ,default='rollout_result_vanilla_mgn.npy')
    return parser.parse_args()

def main():

    args = parse_args()

    param_idx = args.param
    result_file_name = args.result_file_name
    param_idx_ls = [i for i in range(101) if i % 2 ==1]
    data_path = '../data/rawData.npy'
    raw_data = np.load(data_path, allow_pickle=True)
    position_data = np.loadtxt('../data/meshPosition_all.txt')
    node_states = raw_data['x'][param_idx_ls][param_idx]
    node_masses = raw_data['mass']
    edge_attrs = raw_data['edge_attr']
    edge_index = raw_data['edge_index']
    node_positions = position_data.copy()

    # test rollout result
    result_path = '../data/'+result_file_name
    node_states_mgn = np.load(result_path, allow_pickle=True)
    node_states_mgn = np.concatenate((raw_data['x'][param_idx_ls][:,0,:,:][:,np.newaxis,:,:],node_states_mgn ), axis=1)[param_idx]

    # Define figure and axes
    fig, axes = plt.subplots(3,2, figsize=(24, 18))
    titles = ['Ground Truth: u_t','Our Model: u_t','Ground Truth: v_t','Our Model: v_t','Ground Truth: p_t','Our Model: p_t']

    for i, ax in enumerate(axes.flat):
        ax.set_xlim(-1,7)
        ax.set_ylim(-2,2)
        ax.set_title(titles[i])


    # Create grid for interpolation
    grid_x, grid_y = np.meshgrid(
        np.linspace(position_data[:, 0].min()-0.5, position_data[:, 0].max()+0.5, 100),
        np.linspace(position_data[:, 1].min()-0.5, position_data[:, 1].max()+0.5, 100)
    )

    # Function to plot the mesh
    def plot_mesh(ax, data, title):
        interpolated = griddata(position_data, data, (grid_x, grid_y), method='cubic')
        mesh = ax.pcolormesh(grid_x, grid_y, interpolated, shading='auto', cmap=plt.cm.plasma)
        return mesh

    # Initial meshes
    meshes = []
    for i, ax in enumerate(axes.flat):
        if i % 2 == 0:
            mesh = plot_mesh(ax, node_states[0, :, i//2], titles[i])
        else:
            mesh = plot_mesh(ax, node_states_mgn[0, :, i//2], titles[i])
            
        for j in range(edge_index.shape[-1]):
            idx = edge_index[:,j]
            x_1, y_1 = position_data[idx[0]]
            x_2, y_2 = position_data[idx[1]]
            ax.plot(
                [x_1, x_2], [y_1, y_2], color='gray', linestyle='--', linewidth=0.5, alpha=0.8
            )
        meshes.append(mesh)

    def update(frame):
        print(f"This is frame {str(frame)}")
        for i, mesh in enumerate(meshes):
            if i % 2 == 0:
                mesh.set_array(griddata(position_data, node_states[frame, :, i//2], (grid_x, grid_y), method='cubic').ravel())
            else:
                mesh.set_array(griddata(position_data, node_states_mgn[frame, :, i//2], (grid_x, grid_y), method='cubic').ravel())
        return meshes


    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(node_states), interval=100, blit=True)

    # To save the animation as a GIF
    ani.save(f'node_state_time_series_smooth_{str(param_idx)}.gif', writer='Pillow', fps=30)

    plt.show()

if __name__=="__main__":
    main()
