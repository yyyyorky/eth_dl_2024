#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from utils.constant import Constant

C = Constant()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rollout_error_temporal = np.load(os.path.join(C.data_dir, 'result/rollout_error_temporal.npy'))
rollout_error_mgn = np.load(os.path.join(C.data_dir, 'result/rollout_error_mgn.npy'))
# %%
plt.figure()
plt.plot(rollout_error_temporal, label='Temporal')
plt.plot(rollout_error_mgn, label='MGN')
plt.yscale('log')
plt.ylim(1e-2, 2)
plt.legend()
plt.tight_layout()
plt.xlabel('Time step')
plt.ylabel('Relative L2 error')
# %%
