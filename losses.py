import torch
from torch.nn.modules.loss import _Loss


class RMSELoss(_Loss):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self,
                prediction: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        return self.mse(prediction, target).sqrt()


class DiceLoss(torch.nn.Module):
    '''
    Dice loss based on
    https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
    '''
    def forward(self, inputs, targets, smooth=1e-6):
        reduce_dims = (1, 2, 3)
        intersection = (inputs * targets).sum(reduce_dims)
        cardinality = (inputs**2).sum(reduce_dims) + (targets**2).sum(reduce_dims)
        dice = (2*intersection + smooth)/(cardinality + smooth)

        return 1 - dice.mean()


def test_rmse_loss():
    loss = RMSELoss()
    t1 = torch.zeros(1, 1, 256, 256, device='cuda')
    t2 = torch.ones(1, 1, 256, 256, device='cuda')
    print(loss(t1, t1))
    print(loss(t1, t2))


def test_dice_loss():
    loss = DiceLoss()
    t1 = torch.zeros(1, 1, 256, 256, device='cuda')
    t2 = torch.ones(1, 1, 256, 256, device='cuda')
    print(loss(t1, t1))
    print(loss(t1, t2))


if __name__ == '__main__':
    test_dice_loss()
