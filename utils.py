import csv

import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import torch

from reconstruct_interventional import CARMH_GT_UPPER_99_PERCENTILE, \
    hu2mu, mu2hu


def calculate_metrics(prediction: torch.Tensor, groundtruth: torch.Tensor) -> dict[str, float]:
    psnr_val = peak_signal_noise_ratio(
        groundtruth.clamp(0, 1)[None].cpu().numpy(),
        prediction.clamp(0, 1)[None].cpu().numpy(),
        data_range=1.0,
    )

    ssim_val = structural_similarity(
        prediction.clamp(0, 1).cpu().numpy(),
        groundtruth.clamp(0, 1).cpu().numpy(),
        data_range=1.0,
        gaussian_weights=True,
        channel_axis=0,
    )

    # RMSE in HU
    prediction_hu = mu2hu(prediction*hu2mu(CARMH_GT_UPPER_99_PERCENTILE))
    groundtruth_hu = mu2hu(groundtruth*hu2mu(CARMH_GT_UPPER_99_PERCENTILE))
    rmse_val = ((prediction_hu - groundtruth_hu)**2).mean().sqrt()
    return {
        'ssim': ssim_val.item(),
        'psnr': psnr_val.item(),
        'rmse': rmse_val.item(),
    }


def read_csv(path: str, metric: str, reduce: bool = True):
    all_values = []
    with open(path, 'r', encoding='utf-8') as file:
        csvreader = csv.DictReader(file)
        for row in csvreader:
            all_values.append(float(row[metric]))

    if not reduce:
        return all_values

    return np.median(all_values), np.percentile(all_values, 25), \
        np.percentile(all_values, 75)
