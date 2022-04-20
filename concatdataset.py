from functools import cache

import numpy as np
import torch
from torch.utils.data import DataLoader

from singlevolumedataset import SingleVolumeDataset


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.cslen = np.concatenate([
            [0],
            np.cumsum([len(d) for d in datasets])])

    @cache
    def __len__(self):
        return int(self.cslen[-1])

    def __getitem__(self, idx):
        ds_idx = np.searchsorted(self.cslen - 1, idx) - 1
        pos_idx = idx - self.cslen[ds_idx]
        return self.datasets[ds_idx][pos_idx]


def test_dataset():
    volume_path = '/mnt/nvme2/lungs/lungs3d_projections/R_004.pt'
    needle_path = '/home/phernst/Documents/git/ictdl/needle_projections/Needle2_Pos2_12.pt'
    prior_path = '/mnt/nvme2/lungs/lungs3d/priors/R_004.pt'

    dataset = ConcatDataset(
        SingleVolumeDataset(volume_path, needle_path, prior_path, False, True, False, False),
        SingleVolumeDataset(volume_path, needle_path, prior_path, False, True, False, False))
    print(f'{len(dataset)=}')

    _ = dataset[0]
    input_t, gt_t, full_needle_t = dataset[20]
    from matplotlib import pyplot as plt

    dloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=32,
    )
    input_t, gt_t, full_needle_t = next(iter(dloader))
    print(full_needle_t.shape)
    plt.imshow(input_t[20, 0].cpu().numpy())
    plt.figure()
    plt.imshow(input_t[20, 1].cpu().numpy())
    plt.figure()
    plt.imshow(full_needle_t[20, 0].cpu().numpy())
    plt.figure()
    plt.imshow(gt_t[20, 0].cpu().numpy())
    plt.show()


if __name__ == '__main__':
    test_dataset()
