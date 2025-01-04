import argparse
import os
import sys
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.interpolate import griddata

def parse_args():
    parser = argparse.ArgumentParser(description="Smooth Mesh Animation")
    parser.add_argument('--param', type=int, help='Param idx')
    return parser.parse_args()

def main():

    args = parse_args()

    param_idx = args.param
    data_path = '../data/rawData.npy'
    raw_data = np.load(data_path, allow_pickle=True)
    position_data = np.loadtxt('../data/meshPosition_all.txt')
    node_states = raw_data['x'][param_idx]

    # Define figure and axes
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(24, 18))
    for ax, title in zip([ax1, ax2, ax3], ["u_t", "v_t", "p_t"]):
        ax.set_xlim(-1, 7)
        ax.set_ylim(-2, 2)
        ax.set_title(title)

    # Create grid for interpolation
    grid_x, grid_y = np.meshgrid(
        np.linspace(position_data[:, 0].min(), position_data[:, 0].max(), 100),
        np.linspace(position_data[:, 1].min(), position_data[:, 1].max(), 100)
    )

    # Function to plot the mesh
    def plot_mesh(ax, data, title):
        interpolated = griddata(position_data, data, (grid_x, grid_y), method='cubic')
        mesh = ax.pcolormesh(grid_x, grid_y, interpolated, shading='auto', cmap=plt.cm.plasma)
        return mesh

    # Initial meshes
    mesh1 = plot_mesh(ax1, node_states[0, :, 0], "u_t")
    mesh2 = plot_mesh(ax2, node_states[0, :, 1], "v_t")
    mesh3 = plot_mesh(ax3, node_states[0, :, 2], "p_t")

    def update(frame):
        mesh1.set_array(griddata(position_data, node_states[frame, :, 0], (grid_x, grid_y), method='cubic').ravel())
        mesh2.set_array(griddata(position_data, node_states[frame, :, 1], (grid_x, grid_y), method='cubic').ravel())
        mesh3.set_array(griddata(position_data, node_states[frame, :, 2], (grid_x, grid_y), method='cubic').ravel())
        return mesh1, mesh2, mesh3

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(node_states), interval=100, blit=True)

    # To save the animation as a GIF
    ani.save('node_state_time_series_smooth.gif', writer='Pillow', fps=30)

    plt.show()

if __name__=="__main__":
    main()

#node_pos = {i: tuple(position_data[i]) for i in range(len(position_data))}

#nodes_scatter1 = ax1.scatter(position_data[:, 0], position_data[:, 1], c=node_states[0, :, 0], cmap=plt.cm.plasma, s=500)
#nodes_scatter2 = ax2.scatter(position_data[:, 0], position_data[:, 1], c=node_states[0, :, 1], cmap=plt.cm.plasma, s=500)
#nodes_scatter3 = ax3.scatter(position_data[:, 0], position_data[:, 1], c=node_states[0, :, 2], cmap=plt.cm.plasma, s=500)

#def update(frame):
#    nodes_scatter1.set_array(node_states[frame, :, 0])
#    nodes_scatter2.set_array(node_states[frame, :, 1])
#    nodes_scatter3.set_array(node_states[frame, :, 2])
#    return nodes_scatter1, nodes_scatter2, nodes_scatter3

# Create the animation
#ani = FuncAnimation(fig, update, frames=len(node_states), interval=100, blit=True)

# To save the animation as a GIF
#ani.save('node_state_time_series.gif', writer='Pillow', fps=30)

#plt.show()