import argparse
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx
import numpy as np
import json
from matplotlib.animation import FuncAnimation
from scipy.interpolate import griddata

def parse_args():
    parser = argparse.ArgumentParser(description="Smooth Mesh Animation")
    parser.add_argument('--param', type=int, help='Param idx: from 0 to 49',default=0)
    parser.add_argument('--result_file_name', type=str,help="mgn result file name(Please put the result file in the data folder)"
    ,default='rollout_result_vanilla_mgn.npy')
    parser.add_argument('--frame_num', type=int)
    return parser.parse_args()

def add_total_velocity(node_states):
    total_velocity = np.sqrt(node_states[:,:,1]**2 + node_states[:,:,2]**2)[:,:,np.newaxis]
    return np.concatenate((node_states, total_velocity), axis=2)

def main():

    args = parse_args()

    param_idx = args.param
    result_file_name = args.result_file_name
    param_idx_ls = [i for i in range(101) if i % 2 ==1]
    data_path = '../data/rawData.npy'
    raw_data = np.load(data_path,allow_pickle=True)
    position_data = np.loadtxt('../data/meshPosition_all.txt')
    node_states = raw_data['x'][param_idx_ls][param_idx]
    node_states = add_total_velocity(node_states)
    node_masses = raw_data['mass']
    edge_attrs = raw_data['edge_attr']
    edge_index = raw_data['edge_index']
    node_positions = position_data.copy()

    # for training set
    #node_mean = np.mean(node_states,axis=0, keepdims=True)
    #node_std = np.std(node_states, axis=0, keepdims=True)

    # FOR test set
    with open('../data/node_stats.json','r') as f:
        node_stats = json.load(f)
    node_mean = node_stats['node_mean']
    node_std = node_stats['node_std']

    # test rollout result
    result_path = '../data/'+result_file_name
    node_states_mgn = np.load(result_path, allow_pickle=True)*node_std + node_mean
    node_states_mgn = np.concatenate((raw_data['x'][param_idx_ls][:,0,:,:][:,np.newaxis,:,:],node_states_mgn ), axis=1)[param_idx]
    node_states_mgn = add_total_velocity(node_states_mgn)

    # Define figure and axes
    fig, axes = plt.subplots(4, 2, figsize=(24, 24))

    titles = [
        r'Ground Truth: $u_t$', r'MeshGraphNet: $u_t$',
        r'Ground Truth: $v_t$', r'MeshGraphNet: $v_t$',
        r'Ground Truth: $\sqrt{u_t^2 + v_t^2}$', r'MeshGraphNet: $\sqrt{u_t^2 + v_t^2}$',
        r'Ground Truth: $p_t$', r'MeshGraphNet: $p_t$'
    ]


    title_font = {'fontsize': 20, 'fontweight': 'bold', 'family': 'serif'}
    tick_fontsize = 14
    # Customize each subplot
    for i, ax in enumerate(axes.flat):
        ax.set_xlim(-1, 7)
        ax.set_ylim(-2, 2)
        ax.set_title(titles[i], fontdict=title_font) 
        ax.tick_params(axis='both', labelsize=tick_fontsize)  
        ax.set_xticklabels(ax.get_xticks(), fontsize=tick_fontsize, family='serif') 
        ax.set_yticklabels(ax.get_yticks(), fontsize=tick_fontsize, family='serif')


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
    frame_num = args.frame_num if args.frame_num else len(node_states)
    ani = FuncAnimation(fig, update, frames=frame_num, interval=100, blit=True)

    # To save the animation as a GIF
    try:
        ani.save(f'meshgrid/node_state_time_series_mgn.gif', writer='Pillow', fps=30)
    except:
        ani.save(f'node_state_time_series_mgn.gif', writer='Pillow', fps=30)

    plt.show()

if __name__=="__main__":
    main()
