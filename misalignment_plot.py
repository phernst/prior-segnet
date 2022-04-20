from os.path import join as pjoin
from matplotlib import pyplot as plt

import numpy as np

from utils import read_csv


def main():
    misalignment_steps = list(np.linspace(-10, 10, num=101))
    mis_list = [
        read_csv(pjoin(
            'test', 'SegUnet_aug_mis', f"Results{f'_mis{mis_angle}' if mis_angle else ''}.csv"),
            'rmse')
        for mis_angle in misalignment_steps
    ]
    plt.fill_between(
        misalignment_steps,
        [t[1] for t in mis_list],
        [t[2] for t in mis_list],
        alpha=0.2,
    )
    plt.plot(misalignment_steps, [t[0] for t in mis_list], label='Prior-SegNet')

    mis_list = [
        read_csv(pjoin(
            'test', 'NoSegUnet_aug_mis', f"Results{f'_mis{mis_angle}' if mis_angle else ''}.csv"),
            'rmse')
        for mis_angle in misalignment_steps
    ]
    plt.fill_between(
        misalignment_steps,
        [t[1] for t in mis_list],
        [t[2] for t in mis_list],
        alpha=0.2,
    )
    plt.plot(misalignment_steps, [t[0] for t in mis_list], label='Prior-Net')

    mis_list = [
        read_csv(pjoin(
            'test', 'SegUnet', f"Results{f'_mis{mis_angle}' if mis_angle else ''}.csv"),
            'rmse')
        for mis_angle in misalignment_steps
    ]
    plt.fill_between(
        misalignment_steps,
        [t[1] for t in mis_list],
        [t[2] for t in mis_list],
        alpha=0.2,
    )
    plt.plot(misalignment_steps, [t[0] for t in mis_list], label='Prior-SegNet (no mis)')

    plt.xlabel('rotation (deg)')
    plt.ylabel('RMSE (HU)')
    plt.grid()
    # plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
