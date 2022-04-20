from os.path import join as pjoin
from matplotlib import pyplot as plt

from scipy.stats import wilcoxon

from utils import read_csv


def main():
    method_names = {
        'FDK', 'FDKConvNet_aug', 'NoSegNet', 'NoSegUnet_aug_mis',
        'SegUnet', 'SegUnet_aug_mis'
    }

    compare_against = 'SegUnet_aug_mis'
    assert compare_against in method_names

    metric_optimum = {
        'ssim': 'greater',
        'psnr': 'greater',
        'rmse': 'less',
    }
    compare_metric = 'psnr'
    assert compare_metric in metric_optimum

    method_samples = {
        method: read_csv(
            pjoin('test', method, 'Results.csv'),
            compare_metric,
            reduce=False)
        for method in method_names
    }

    method_pvalues = {
        method: wilcoxon(
            method_samples[compare_against],
            method_samples[method],
            alternative=metric_optimum[compare_metric],
            ).pvalue
        for method in method_names if method != compare_against
    }
    print(f'{method_pvalues=}')

    significance_level = 0.005
    pairwise_significant = all(
        pval < significance_level
        for pval in method_pvalues.values()
    )
    print(f'{pairwise_significant=}')
    if not pairwise_significant:
        print(f'{method_pvalues=}')

    for method, samples in method_samples.items():
        plt.hist(samples, bins=10, histtype='step', label=method)
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
