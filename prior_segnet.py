from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F


class PriorSegNet(nn.Module):
    def __init__(self, wf=3, batch_norm=True):
        super().__init__()

        sfs = 2**wf

        # prior
        self.conv_prior_1 = SegNetDownBlock(1, (sfs, sfs * 2), batch_norm)  # 8, 16
        self.conv_prior_2 = SegNetDownBlock(sfs*2, (sfs * 2, sfs * 4), batch_norm)  # 16, 32
        self.conv_prior_3 = SegNetDownBlock(sfs*4, (sfs * 4, sfs * 8), batch_norm)  # 32, 64
        self.conv_prior_4 = SegNetDownBlock(sfs*8, (sfs * 8, sfs * 16), batch_norm)  # 64, 128
        self.conv_prior_5 = SegNetDownBlock(sfs*16, (sfs * 16, sfs * 32), batch_norm)  # 128, 256

        # cbct
        self.conv_cbct_1 = SegNetDownBlock(1, (sfs, sfs * 2), batch_norm)  # 8, 16
        self.conv_cbct_2 = SegNetDownBlock(sfs*2, (sfs * 2, sfs * 4), batch_norm)  # 16, 32
        self.conv_cbct_3 = SegNetDownBlock(sfs*4, (sfs * 4, sfs * 8), batch_norm)  # 32, 64
        self.conv_cbct_4 = SegNetDownBlock(sfs*8, (sfs * 8, sfs * 16), batch_norm)  # 64, 128
        self.conv_cbct_5 = SegNetDownBlock(sfs*16, (sfs * 16, sfs * 32), batch_norm)  # 128, 256

        # deconv
        self.deconv1 = SegNetUpBlock(sfs*64, (sfs*64, sfs*32), batch_norm)
        self.deconv2 = SegNetUpBlock(sfs*64, (sfs*16, sfs*16), batch_norm)
        self.deconv3 = SegNetUpBlock(sfs*32, (sfs*8, sfs*8), batch_norm)
        self.deconv4 = SegNetUpBlock(sfs*16, (sfs*4, sfs*4), batch_norm)
        self.deconv5 = SegNetUpBlock(sfs*8, (sfs*2, sfs*2), batch_norm)

        # conv level 0
        self.conv_level0 = [
            nn.Conv2d(sfs*2, sfs, 3, padding=1),
            nn.LeakyReLU(0.2),
        ]
        if batch_norm:
            self.conv_level0.append(nn.BatchNorm2d(sfs))
        self.conv_level0 += [
            nn.Conv2d(sfs, sfs*4, 3, padding=1),
            nn.LeakyReLU(0.2),
        ]
        if batch_norm:
            self.conv_level0.append(nn.BatchNorm2d(sfs*4))
        self.conv_level0 = nn.Sequential(*self.conv_level0)

        # outputs
        self.reco_out = nn.Sequential(
            nn.Conv2d(sfs*4, 1, 3, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.seg_out = nn.Sequential(
            nn.Conv2d(sfs*4, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        prior = x[:, 0:1]
        cbct = x[:, 1:2]

        prior1 = self.conv_prior_1(prior)
        prior2 = self.conv_prior_2(prior1)
        prior3 = self.conv_prior_3(prior2)
        prior4 = self.conv_prior_4(prior3)
        prior5 = self.conv_prior_5(prior4)

        cbct1 = self.conv_cbct_1(cbct)
        cbct2 = self.conv_cbct_2(cbct1)
        cbct3 = self.conv_cbct_3(cbct2)
        cbct4 = self.conv_cbct_4(cbct3)
        cbct5 = self.conv_cbct_5(cbct4)

        deconv1 = self.deconv1(prior5, cbct5)
        deconv2 = self.deconv2(deconv1, prior4, cbct4)
        deconv3 = self.deconv3(deconv2, prior3, cbct3)
        deconv4 = self.deconv4(deconv3, prior2, cbct2)
        deconv5 = self.deconv5(deconv4, prior1, cbct1)

        deconv5 = SegNetUpBlock.embed_layer(deconv5, x.shape[2:])

        conv_level0 = self.conv_level0(deconv5)

        return self.reco_out(conv_level0), self.seg_out(conv_level0)


class SegNetDownBlock(nn.Module):
    def __init__(self, in_size, filter_size: Tuple[int, int], batch_norm):
        super().__init__()
        block = []

        block.append(nn.Conv2d(in_size, filter_size[0], kernel_size=3,
                               padding=1))
        block.append(nn.LeakyReLU(0.2))
        if batch_norm:
            block.append(nn.BatchNorm2d(filter_size[0]))

        block.append(nn.Conv2d(filter_size[0], filter_size[1], kernel_size=3,
                               stride=2, padding=1))
        block.append(nn.LeakyReLU(0.2))
        if batch_norm:
            block.append(nn.BatchNorm2d(filter_size[1]))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class SegNetUpBlock(nn.Module):
    def __init__(self, in_size, filter_size: Tuple[int, int], batch_norm):
        super().__init__()
        block = []

        block.append(nn.Conv2d(
            in_size, filter_size[0], kernel_size=3, padding=1))
        block.append(nn.LeakyReLU(0.2))
        if batch_norm:
            block.append(nn.BatchNorm2d(filter_size[0]))

        block.append(nn.ConvTranspose2d(
            filter_size[0], filter_size[1], kernel_size=3,
            stride=2, padding=1))
        block.append(nn.LeakyReLU(0.2))
        if batch_norm:
            block.append(nn.BatchNorm2d(filter_size[1]))

        self.block = nn.Sequential(*block)

    @staticmethod
    def embed_layer(layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_x_0 = (target_size[1] - layer_width) // 2
        diff_x_1 = target_size[1] - layer_width - diff_x_0
        diff_y_0 = (target_size[0] - layer_height) // 2
        diff_y_1 = target_size[0] - layer_height - diff_y_0
        layer = F.pad(layer, (diff_x_0, diff_x_1, diff_y_0, diff_y_1))
        return layer

    def forward(self, *args):
        args = [SegNetUpBlock.embed_layer(k, args[-1].shape[2:]) for k in args]
        up = torch.cat(args, 1)
        up = self.block(up)

        return up


if __name__ == '__main__':
    from torchinfo import summary
    model = PriorSegNet().cuda()
    summary(model, (1, 2, 384, 384))
