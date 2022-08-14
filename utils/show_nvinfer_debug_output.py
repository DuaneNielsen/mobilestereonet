
import numpy as np
from matplotlib import pyplot as plt

fix, ax = plt.subplots(8)

seq_id = 0

for seq_id in range(0, 20):
    ax[seq_id%4*2].clear()
    ax[seq_id%4*2+1].clear()

    left = np.fromfile(
        f'../gstnvdsinfer_uid-01_layer-left_batch-{seq_id:010}_batchsize-01.bin',
        dtype=np.float32).reshape(3, 384, 1248*2).transpose(1, 2, 0)[:, :1248, :]

    # load nvinfer input from file and verify its the same
    disparity = np.fromfile(
        f'../gstnvdsinfer_uid-01_layer-disparity_batch-{seq_id:010}_batchsize-01.bin',
        dtype=np.float32).reshape(384, 1248)

    #disparity = 1.0/disparity

    ax[seq_id%4*2].imshow(left)
    ax[seq_id%4*2+1].imshow((disparity - np.min(disparity)) / np.max(disparity), cmap='viridis')
    plt.pause(1.0)


plt.show()