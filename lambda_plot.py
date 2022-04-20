from os.path import join as pjoin

from matplotlib import pyplot as plt
import numpy as np

from utils import read_csv


def main():
    lambda_to_name = {
        1e-5: 'SegNet_aug_mis_r1_s1e-5',
        1e-4: 'SegNet_aug_mis_r1_s1e-4',
        1e-3: 'SegUnet_aug_mis',
        1e-2: 'SegNet_aug_mis_r1_s1e-2',
        1e-1: 'SegNet_aug_mis_r1_s1e-1',
    }

    lambda_metrics = {
        k: read_csv(pjoin('test', v, 'Results.csv'), 'rmse')
        for (k, v) in lambda_to_name.items()
    }

    lambda_list = sorted(list(lambda_to_name.keys()))

    lambda_median = [lambda_metrics[l][0] for l in lambda_list]
    lambda_q1 = [lambda_metrics[l][1] for l in lambda_list]
    lambda_q3 = [lambda_metrics[l][2] for l in lambda_list]

    # plt.fill_between(lambda_list, lambda_q1, lambda_q3, alpha=0.2)
    # plt.plot(lambda_list, lambda_median)
    # plt.xscale('log')
    # plt.show()

    lambda_errors = np.stack(
        (
            [m - q for m, q in zip(lambda_median, lambda_q1)],
            [q - m for m, q in zip(lambda_median, lambda_q3)],
        ),
        axis=0,
    )

    plt.errorbar(lambda_list, lambda_median, yerr=lambda_errors, marker='.', capsize=5)
    plt.xlabel('$\\lambda$')
    plt.ylabel('RMSE (HU)')
    plt.xscale('log')
    plt.grid()
    # plt.ylim(bottom=0)
    plt.show()


if __name__ == '__main__':
    main()
