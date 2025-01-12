#%% Run this script to plot the relative L2 error of the four methods
# If you want to produce the error history by yourself, please run the corresponding script
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from utils.constant import Constant

C = Constant()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rollout_error_temporal_2mp = np.load(os.path.join(C.data_dir, 'result/rollout_error_temporal_2mp.npy'))
rollout_error_mgn = np.load(os.path.join(C.data_dir, 'result/rollout_error_mgn.npy'))
rollout_error_temporal_mp = np.load(os.path.join(C.data_dir, 'result/rollout_error_temporal_mp.npy'))
rollout_error_temporal_spatial = np.load(os.path.join(C.data_dir, 'result/rollout_error_temporal_spatial.npy'))
# %%
plt.figure()
plt.plot(rollout_error_mgn, label='MGN')
plt.plot(rollout_error_temporal_mp, label='Temporal-MP')
plt.plot(rollout_error_temporal_2mp, label='Temporal-2MP')
plt.plot(rollout_error_temporal_spatial, label='Temporal-Spatial')
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.xlabel('Time step')
plt.ylabel('Relative L2 error')
# %%