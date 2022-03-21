from argparse import ArgumentParser
from typing import List, Any
import json
import os
from os.path import join as pjoin

import cv2
from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from concatdataset import ConcatDataset
from singlevolumedataset import SingleVolumeDataset

from prior_segunet import UNet
from losses import RMSELoss, DiceLoss


class PriorSegUnet(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.valid_dir = self.hparams.valid_dir
        self.network = UNet()

        self.reco_loss = F.mse_loss
        self.seg_loss = DiceLoss()
        self.accuracy = RMSELoss()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
        )
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     max_lr=self.hparams.lr,
        #     epochs=self.hparams.max_epochs,
        #     steps_per_epoch=len(self.train_dataloader()),
        # )
        return {
            'optimizer': optimizer,
            # 'lr_scheduler': {
            #     'scheduler': scheduler,
            #     'interval': 'step',
            # },
            'monitor': 'val_loss',
        }

    def forward(self, *args, **kwargs):
        net_in = args[0]
        return self.network(net_in)

    def training_step(self, *args, **kwargs):
        batch = args[0]
        batch_in, gt_reco, gt_seg = batch
        prediction = self(batch_in)
        reco_loss = self.reco_loss(prediction[0], gt_reco)
        seg_loss = self.seg_loss(prediction[1], gt_seg)
        loss = reco_loss + 1e-3*seg_loss
        return {
            'loss': loss,
            'reco_loss': reco_loss.detach(),
            'seg_loss': seg_loss.detach(),
        }

    def validation_step(self, *args, **kwargs):
        batch, batch_idx = args[0], args[1]
        batch_in, gt_reco, gt_seg = batch
        prediction = self(batch_in)
        reco_loss = self.reco_loss(prediction[0], gt_reco)
        seg_loss = self.seg_loss(prediction[1], gt_seg)
        loss = reco_loss + 1e-3*seg_loss
        accuracy = self.accuracy(prediction[0], gt_reco)

        if batch_idx < 20:
            os.makedirs(
                pjoin(
                    self.valid_dir, self.hparams.run_name,
                    f'{self.current_epoch}'
                ),
                exist_ok=True,
            )
            gt_reco = gt_reco.cpu().numpy()[0, 0]
            cv2.imwrite(
                pjoin(self.valid_dir, self.hparams.run_name,
                      f'{self.current_epoch}/{batch_idx}_out_gt.png'),
                gt_reco/0.7*255)

            prediction = prediction[0].cpu().float().numpy()[0, 0]
            cv2.imwrite(
                pjoin(self.valid_dir, self.hparams.run_name,
                      f'{self.current_epoch}/{batch_idx}_out_pred.png'),
                prediction/0.7*255,
            )

        return {
            'val_loss': loss,
            'val_reco_loss': reco_loss,
            'val_seg_loss': seg_loss,
            'val_acc': accuracy,
        }

    def create_dataset(self, subject_path: str, needle_path: str,
                       prior_path: str, train: bool) -> SingleVolumeDataset:
        return SingleVolumeDataset(
            subject_path,
            needle_path,
            prior_path,
            shuffle=train,
            augment_needle=train,
        )

    def train_dataloader(self) -> DataLoader:
        full_dataset = ConcatDataset(*[
            self.create_dataset(
                pjoin(self.hparams.subject_dir, f'{f}.pt'),
                pjoin(self.hparams.needle_dir, n),
                pjoin(self.hparams.prior_dir, f'{f}.pt'),
                train=True,
            )
            for f in self.hparams.train_subjects
            for n in os.listdir(self.hparams.needle_dir)
            if n.endswith('.pt')
        ])

        return DataLoader(
            full_dataset,
            shuffle=False,
            batch_size=self.hparams.batch_size,
        )

    def val_dataloader(self) -> DataLoader:
        full_dataset = ConcatDataset(*[
            self.create_dataset(
                pjoin(self.hparams.subject_dir, f'{f}.pt'),
                pjoin(self.hparams.needle_dir, n),
                pjoin(self.hparams.prior_dir, f'{f}.pt'),
                train=False,
            )
            for f in self.hparams.valid_subjects
            for n in os.listdir(self.hparams.needle_dir)
            if n.endswith('.pt') and ('Part' not in n)
        ])

        return DataLoader(
            full_dataset,
            shuffle=False,
            batch_size=self.hparams.batch_size,
        )

    def predict_dataloader(self):
        ...

    def test_dataloader(self):
        ...

    def training_epoch_end(self, outputs: List[Any]) -> None:
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_reco_loss = torch.stack([x['reco_loss'] for x in outputs]).mean()
        avg_seg_loss = torch.stack([x['seg_loss'] for x in outputs]).mean()
        self.log('loss', avg_loss)
        self.log('reco_loss', avg_reco_loss)
        self.log('seg_loss', avg_seg_loss)

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_reco_loss = torch.stack([x['val_reco_loss'] for x in outputs]).mean()
        avg_seg_loss = torch.stack([x['val_seg_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)
        self.log('val_reco_loss', avg_reco_loss)
        self.log('val_seg_loss', avg_seg_loss)
        self.log('val_acc', avg_acc)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--run_name', type=str)
        parser.add_argument('--subject_dir', type=str)
        parser.add_argument('--needle_dir', type=str)
        parser.add_argument('--prior_dir', type=str)
        parser.add_argument('--valid_dir', type=str)
        parser.add_argument('--train_subjects', type=list)
        parser.add_argument('--valid_subjects', type=list)
        return parser


def main():
    seed_everything(42)

    parser = ArgumentParser()
    parser = PriorSegUnet.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    hparams.use_amp = True
    hparams.lr = 1e-3
    hparams.max_epochs = 150
    hparams.batch_size = 32
    hparams.run_name = 'SegUnet'
    hparams.valid_dir = 'valid'
    hparams.subject_dir = '/mnt/nvme2/lungs/lungs3d_projections'
    hparams.needle_dir = '/home/phernst/Documents/git/ictdl/needle_projections'
    hparams.prior_dir = '/mnt/nvme2/lungs/lungs3d/priors'
    with open('train_valid_test.json', 'r', encoding='utf-8') as json_file:
        json_dict = json.load(json_file)
        hparams.train_subjects = json_dict['train_subjects']
        hparams.valid_subjects = json_dict['valid_subjects']

    model = PriorSegUnet(**vars(hparams))

    checkpoint_callback = ModelCheckpoint(
        dirpath=pjoin(hparams.valid_dir, hparams.run_name),
        monitor='val_loss',
        save_last=True,
    )
    lr_callback = LearningRateMonitor()
    logger = TensorBoardLogger('lightning_logs', name=hparams.run_name)

    trainer = Trainer(
        logger=logger,
        precision=16 if hparams.use_amp else 32,
        gpus=1,
        callbacks=[checkpoint_callback, lr_callback],
        max_epochs=hparams.max_epochs,
    )
    trainer.fit(model)


if __name__ == '__main__':
    main()
