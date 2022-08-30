import json
import os
from os.path import join as pjoin
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd
from pytorch_lightning import seed_everything
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from concatdataset import ConcatDataset
from singlevolumedataset import SingleVolumeDataset
from train_fdkconv import FDKConvNet
from utils import calculate_metrics


def main(run_name: str, save_visuals: bool = False,
         photon_flux: Optional[int] = None):
    with open('train_valid_test.json', 'r', encoding='utf-8') as json_file:
        test_subjects = json.load(json_file)['test_subjects']

    checkpoint_dir = pjoin('valid', run_name)
    checkpoint_path = sorted([x for x in os.listdir(checkpoint_dir) if "epoch" in x])[-1]

    out_dir = pjoin('test', run_name)
    os.makedirs(out_dir, exist_ok=True)

    visual_dir = pjoin('visual', run_name)
    os.makedirs(visual_dir, exist_ok=True)

    model = FDKConvNet.load_from_checkpoint(
        pjoin(checkpoint_dir, checkpoint_path))
    model.eval()
    model.cuda()

    test_dataset = ConcatDataset(*[
        SingleVolumeDataset(
            pjoin(model.hparams.subject_dir, f'{f}.pt'),
            pjoin(model.hparams.needle_dir, n),
            pjoin(model.hparams.prior_dir, f'{f}.pt'),
            shuffle=False,
            augment_needle=False,
            augment_all=False,
            misalign=False,
            photon_flux=10**photon_flux if photon_flux is not None else None,
        )
        for f in test_subjects
        for n in os.listdir(model.hparams.needle_dir)
        if n.endswith('.pt') and ('Part' not in n)
    ])

    dataloader_test = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=model.hparams.batch_size,
    )

    metrics = []
    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(dataloader_test)):
            batch_in, batch_gt_reco, _ = batch
            batch_prediction = model(batch_in)

            for prediction, gt_reco in zip(batch_prediction, batch_gt_reco):
                metrics.append(calculate_metrics(prediction, gt_reco))

            if save_visuals and batch_idx == 0:
                img = nib.Nifti1Image(batch_gt_reco[:, 0].cpu().numpy().transpose(), np.eye(4))
                nib.save(img, pjoin(visual_dir, "gt.nii.gz"))
                img = nib.Nifti1Image(batch_in[:, 0].cpu().numpy().transpose(), np.eye(4))
                nib.save(img, pjoin(visual_dir, "in.nii.gz"))
                img = nib.Nifti1Image(batch_in[:, 1].cpu().numpy().transpose(), np.eye(4))
                nib.save(img, pjoin(visual_dir, "prior.nii.gz"))
                img = nib.Nifti1Image(batch_prediction[:, 0].cpu().numpy().transpose(), np.eye(4))
                nib.save(img, pjoin(visual_dir, "prediction.nii.gz"))

    df = pd.DataFrame.from_dict(metrics)
    df.to_csv(pjoin(out_dir, f"Results{f'_noise{photon_flux}' if photon_flux is not None else ''}.csv"))


def create_visuals(run_name: str):
    main(run_name, save_visuals=True)


def run_photon_flux_sweep(run_name: str):
    out_dir = pjoin('test', run_name)
    photon_flux_steps = np.linspace(2, 6, num=5)
    for photon_flux in tqdm(photon_flux_steps):
        if os.path.exists(pjoin(out_dir, f"Results_noise{photon_flux}.csv")):
            continue
        main(run_name, photon_flux=photon_flux)


if __name__ == '__main__':
    seed_everything(42)
    main('FDKConvNet_aug')
    # create_visuals('FDKConvNet_aug')
    # run_photon_flux_sweep("FDKConvNet_aug")
