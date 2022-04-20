import json
import os
from os.path import join as pjoin

import pandas as pd
from pytorch_lightning import seed_everything
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from concatdataset import ConcatDataset
from singlevolumedataset import SingleVolumeDataset
from train_segnet import PriorSegNet
from utils import calculate_metrics


def main():
    with open('train_valid_test.json', 'r', encoding='utf-8') as json_file:
        test_subjects = json.load(json_file)['test_subjects']

    checkpoint_dir = pjoin('valid', 'SegUnet_aug_mis')
    checkpoint_path = sorted([x for x in os.listdir(checkpoint_dir) if "epoch" in x])[-1]

    out_dir = pjoin('test', 'FDK')
    os.makedirs(out_dir, exist_ok=True)

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
        for batch in tqdm(dataloader_test):
            batch_in, batch_gt_reco, _ = batch

            for prediction, gt_reco in zip(batch_in[:, :1], batch_gt_reco):
                metrics.append(calculate_metrics(prediction, gt_reco))

    df = pd.DataFrame.from_dict(metrics)
    df.to_csv(pjoin(out_dir, 'Results.csv'))


if __name__ == '__main__':
    seed_everything(42)
    main()
