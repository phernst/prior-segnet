from functools import cache
import random
from typing import Tuple

import numpy as np
import torch
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode

from reconstruct_interventional import create_network_data, hu2mu


class SingleVolumeDataset(torch.utils.data.Dataset):
    AUGMENTATION_MAX_ANGLE: float = 180.0
    MAX_ANGLE_PRIOR: float = 5.0

    def __init__(self, volume_path: str,
                 needle_projections_path: str, prior_path: str,
                 shuffle: bool,
                 augment_needle: bool,
                 augment_all: bool,
                 misalign: bool):
        self.volume_path = volume_path
        self.needle_projections_path = needle_projections_path
        self.prior_path = prior_path
        self.reco_tensor = None
        self.shuffle = shuffle
        self.augment_needle = augment_needle
        self.augment_all = augment_all
        self.misalign = misalign
        self.running_indices = list(range(len(self)))

    def _reconstruct(self) -> torch.Tensor:
        vol_ds = torch.load(self.volume_path)
        volume_shape = list(vol_ds['volume_shape'])
        needle_projections = torch.load(self.needle_projections_path)
        if self.augment_needle:
            needle_projections = needle_projections.roll(
                random.randint(0, 359), -1)
        prior_ds = hu2mu(torch.load(self.prior_path)['volume'][..., None])
        prior_ds[prior_ds < 0] = 0
        return create_network_data(
            (vol_ds['projections'], vol_ds['voxel_size'], volume_shape),
            needle_projections,
            prior_ds,
        )

    @cache
    def __len__(self):
        return self._reconstruct().shape[0]

    def _augmentation(self, input_t: torch.Tensor, groundtruth_t: torch.Tensor,
                      full_needle_t: torch.Tensor, misalign: bool):
        # 0-none 1-rotate 2-scale 3-flip 4-fliprotate
        rand_num = random.randint(0, 4)
        scale_ratio = [0.8, 1.2]

        if rand_num == 1:
            input_t, groundtruth_t, full_needle_t = self._rotate(
                input_t,
                groundtruth_t,
                full_needle_t,
                angle=self.AUGMENTATION_MAX_ANGLE,
            )
        if rand_num == 2:
            input_t, groundtruth_t, full_needle_t = self._scale(
                input_t,
                groundtruth_t,
                full_needle_t,
                scales=scale_ratio,
            )
        if rand_num == 3:
            input_t, groundtruth_t, full_needle_t = self._flip(
                input_t,
                groundtruth_t,
                full_needle_t,
            )
        if rand_num == 4:
            input_t, groundtruth_t, full_needle_t = self._rotate(
                *self._flip(
                    input_t,
                    groundtruth_t,
                    full_needle_t
                ),
                angle=self.AUGMENTATION_MAX_ANGLE,
            )

        if misalign:
            if random.random() > 0.5:
                input_t[1:2] = self._rotate(
                    input_t[1:2],
                    angle=self.MAX_ANGLE_PRIOR,
                )[0]

        return input_t, groundtruth_t, full_needle_t

    @staticmethod
    def _rotate(*args, angle: float):
        rot_angle = random.uniform(-angle, angle)
        return [F.rotate(a, rot_angle, interpolation=InterpolationMode.BILINEAR) for a in args]

    @staticmethod
    def _scale(*args, scales: Tuple[float, float]):
        rnd_scale = random.uniform(scales[0], scales[1])
        return [F.affine(
            a,
            angle=0,
            translate=(0, 0),
            scale=rnd_scale,
            shear=0,
            interpolation=InterpolationMode.BILINEAR) for a in args]

    @staticmethod
    def _flip(*args):
        return [a.fliplr() for a in args]

    def __getitem__(self, idx):
        if idx == 0:
            self.reco_tensor = self._reconstruct()
            if self.shuffle:
                np.random.shuffle(self.running_indices)

        full_data = self.reco_tensor[self.running_indices[idx]]
        if idx == len(self) - 1:
            full_data = torch.clone(full_data)
            self.reco_tensor = None

        input_t = torch.stack((full_data[..., 0], full_data[..., 3]))  # [2, w, h]
        gt_t = full_data[None, ..., 1]
        full_needle_t = full_data[None, ..., 2]

        if self.augment_all:
            input_t, gt_t, full_needle_t = self._augmentation(
                input_t,
                gt_t,
                full_needle_t,
                misalign=self.misalign,
            )

        input_t, (gt_t, full_needle_t) = _binarize_needle(
            input_t, (gt_t, full_needle_t)
        )

        return input_t, gt_t, full_needle_t


def needle_segmentation(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor > 0.7).to(tensor.dtype)


def _binarize_needle(tensor_in: torch.Tensor, tensor_out: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    segmentation_tensor = needle_segmentation(tensor_out[1])
    return tensor_in, (tensor_out[0], segmentation_tensor)


def test_dataset():
    volume_path = '/mnt/nvme2/lungs/lungs3d_projections/R_004.pt'
    needle_path = '/home/phernst/Documents/git/ictdl/needle_projections/Needle2_Pos2_12.pt'
    prior_path = '/mnt/nvme2/lungs/lungs3d/priors/R_004.pt'

    dataset = SingleVolumeDataset(
        volume_path,
        needle_path,
        prior_path,
        False,
        True,
        True,
        True,
    )
    print(f'{len(dataset)=}')

    _ = dataset[0]
    input_t, gt_t, full_needle_t = dataset[20]
    from matplotlib import pyplot as plt
    plt.imshow(input_t[0].cpu().numpy())
    plt.figure()
    plt.imshow(input_t[1].cpu().numpy())
    plt.figure()
    plt.imshow(full_needle_t[0].cpu().numpy())
    plt.figure()
    plt.imshow(gt_t[0].cpu().numpy())
    plt.show()


if __name__ == '__main__':
    test_dataset()
