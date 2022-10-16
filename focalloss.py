import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, size_average=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        input = input.view(-1, 1)
        target = target.view(-1, 1)

        first = -self.alpha[0] * target * (1 - input) ** self.gamma * torch.log(input)
        second = -self.alpha[1] * (1 - target) * (1 - input) ** self.gamma * torch.log(input)
        loss = first + second

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


if __name__ == "__main__":
    x = torch.sigmoid(torch.randn(1, 3, 50, 50))
    y = (torch.randn(1, 3, 50, 50) >= 0).float()

    loss = FocalLoss(gamma=10)
    loss(x, y)
    print(x.shape)
