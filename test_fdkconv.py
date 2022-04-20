import json
import os
from os.path import join as pjoin

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


def main(run_name: str, save_visuals: bool = False):
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
        for batch_idx, batch in tqdm(enumerate(dataloader_test)):
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
    df.to_csv(pjoin(out_dir, "Results.csv"))


def create_visuals(run_name: str):
    main(run_name, save_visuals=True)


if __name__ == '__main__':
    seed_everything(42)
    # main('FDKConvNet_aug')
    create_visuals('FDKConvNet_aug')
