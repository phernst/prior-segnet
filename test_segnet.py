import json
import os
from os.path import join as pjoin
from typing import Optional, Union

import nibabel as nib
import numpy as np
import pandas as pd
from pytorch_lightning import seed_everything
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from concatdataset import ConcatDataset
from singlevolumedataset import SingleVolumeDataset
from train_segnet import PriorSegNet
from utils import calculate_metrics


def main(run_name: str, misalign: Union[bool, float] = False,
         save_visuals: bool = False, photon_flux: Optional[int] = None):
    with open('train_valid_test.json', 'r', encoding='utf-8') as json_file:
        test_subjects = json.load(json_file)['test_subjects']

    checkpoint_dir = pjoin('valid', run_name)
    checkpoint_path = sorted([x for x in os.listdir(checkpoint_dir) if "epoch" in x])[-1]

    out_dir = pjoin('test', run_name)
    os.makedirs(out_dir, exist_ok=True)

    visual_dir = pjoin('visual', run_name)
    os.makedirs(visual_dir, exist_ok=True)

    model = PriorSegNet.load_from_checkpoint(
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
            misalign=misalign,
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
            batch_prediction = model(batch_in)[0]

            for prediction, gt_reco in zip(batch_prediction, batch_gt_reco):
                metrics.append(calculate_metrics(prediction, gt_reco))

            if save_visuals and batch_idx == 0:
                img = nib.Nifti1Image(batch_gt_reco[:, 0].cpu().numpy().transpose(), np.eye(4))
                nib.save(img, pjoin(visual_dir, f"gt{f'_rot{misalign}' if misalign else ''}.nii.gz"))
                img = nib.Nifti1Image(batch_in[:, 0].cpu().numpy().transpose(), np.eye(4))
                nib.save(img, pjoin(visual_dir, f"in{f'_rot{misalign}' if misalign else ''}.nii.gz"))
                img = nib.Nifti1Image(batch_in[:, 1].cpu().numpy().transpose(), np.eye(4))
                nib.save(img, pjoin(visual_dir, f"prior{f'_rot{misalign}' if misalign else ''}.nii.gz"))
                img = nib.Nifti1Image(batch_prediction[:, 0].cpu().numpy().transpose(), np.eye(4))
                nib.save(img, pjoin(visual_dir, f"prediction{f'_rot{misalign}' if misalign else ''}.nii.gz"))
                break

    df = pd.DataFrame.from_dict(metrics)
    df.to_csv(pjoin(out_dir, f"Results{f'_mis{misalign}' if misalign else ''}{f'_noise{photon_flux}' if photon_flux is not None else ''}.csv"))


def create_visuals(run_name: str, misalign: Union[bool, float]):
    main(run_name, misalign, save_visuals=True)


def run_misalign_sweep(run_name: str):
    out_dir = pjoin('test', run_name)
    misalignment_steps = np.linspace(-10, 10, num=101)
    for misalignment_angle in tqdm(misalignment_steps):
        if os.path.exists(pjoin(out_dir, f"Results{f'_mis{misalignment_angle}' if misalignment_angle else ''}.csv")):
            continue
        main(run_name, misalign=misalignment_angle)


def run_photon_flux_sweep(run_name: str):
    out_dir = pjoin('test', run_name)
    photon_flux_steps = np.linspace(2, 6, num=5)
    for photon_flux in tqdm(photon_flux_steps):
        if os.path.exists(pjoin(out_dir, f"Results_noise{photon_flux}.csv")):
            continue
        main(run_name, photon_flux=photon_flux)


if __name__ == '__main__':
    seed_everything(42)
    # main('NoSegUnet_aug_mis')
    # run_misalign_sweep('SegUnet_aug_mis')
    create_visuals('SegUnet_aug_mis', 0.0)
    # run_photon_flux_sweep('SegUnet_aug_mis')

    # main('NoSegUnet_aug_mis')
    # main('SegNet_aug_mis_r1_s1e-5')
    # main('SegNet_aug_mis_r1_s1e-4')
    # main('SegUnet_aug_mis')
    # main('SegNet_aug_mis_r1_s1e-2')
    # main('SegNet_aug_mis_r1_s1e-1')
    # main('SegUnet')
