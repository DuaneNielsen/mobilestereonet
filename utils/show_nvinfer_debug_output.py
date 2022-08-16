import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

fix, ax = plt.subplots(3)

seq_id = 0


def read_all_lines(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    return lines

path = '/mnt/sdd2/home/duane/data/kitti2015'

lines = read_all_lines('../filenames/kitti15_train.txt')
splits = [line.split() for line in lines]
disp_images = [x[2] for x in splits]

for seq_id in range(0, 20):
    ax[seq_id % 1 * 2].clear()
    ax[seq_id % 1 * 2 + 1].clear()

    image = np.fromfile(
        f'../gstnvdsinfer_uid-01_layer-left_batch-{seq_id:010}_batchsize-01.bin',
        dtype=np.float32).reshape(3, 384, 1248 * 2).transpose(1, 2, 0)
    left = image[:, :1248, :]
    right = image[:, 1248:, :]

    # load nvinfer input from file and verify its the same
    disparity = np.fromfile(
        f'../gstnvdsinfer_uid-01_layer-disparity_batch-{seq_id:010}_batchsize-01.bin',
        dtype=np.float32).reshape(384, 1248)

    disparity_gt = Image.open(f'{path}/{disp_images[seq_id]}')

    # disparity = 1.0/disparity

    ax[seq_id % 1 * 2].imshow(left)
    ax[seq_id % 1 * 2 + 1].imshow((disparity - np.min(disparity)) / np.max(disparity), cmap='plasma')
    ax[2].imshow(disparity_gt)
    plt.pause(1.0)

plt.show()
