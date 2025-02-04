from pathlib import Path
from argparse import ArgumentParser
from datasets import __datasets__
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', help='dataset name', choices=__datasets__.keys())
    parser.add_argument('--datapath', required=True, help='data path')
    parser.add_argument('--filelist', required=True, help='file list of left, right pairs')
    parser.add_argument('--savepath', required=True, help='directory to save dataset to')
    args = parser.parse_args()

    StereoDataset = __datasets__[args.dataset]
    test_dataset = StereoDataset(args.datapath, args.filelist, False, raw=True)
    TestImgLoader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    for batch_idx, sample in enumerate(tqdm(TestImgLoader)):
        left, right = sample['left'], sample['right']
        image = torch.cat((left, right), axis=3)
        im = Image.fromarray(image[0].numpy().transpose(1, 2, 0))
        im.save(f'{args.savepath}/frame_{batch_idx:05}.png')

