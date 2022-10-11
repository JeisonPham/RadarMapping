import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_group(cin, cout, kernel_size, dilation=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=kernel_size, padding=kernel_size // 2, dilation=dilation),
        nn.GroupNorm(cout, cout),
        nn.ReLU()
    )


class MapBackBone(nn.Module):
    def __init__(self, cin, H=256, W=256):
        super(MapBackBone, self).__init__()

        self.layer1 = nn.Sequential(
            generate_group(cin, 32, 3),
            generate_group(32, 32, 3)
        )
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(3, stride=2, padding=1),
            generate_group(32, 64, 3)
        )
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(3, stride=2, padding=1),
            generate_group(64, 128, 3),
            generate_group(128, 128, 3)
        )
        self.layer4 = nn.Sequential(
            nn.MaxPool2d(3, stride=2, padding=1),
            generate_group(128, 256, 3),
            generate_group(256, 256, 3),
            generate_group(256, 256, 3),
            generate_group(256, 256, 3),
            generate_group(256, 256, 3)
        )

        self.final = nn.Sequential(
            generate_group(32 + 64 + 128 + 256, 256, 3),
            generate_group(256, 256, 3),
            generate_group(256, 256, 3),
            generate_group(256, 256, 3)
        )

        self.avgpool1 = nn.AvgPool2d(3, stride=2, padding=1)
        self.avgpool2 = nn.AvgPool2d(3, stride=2, padding=1)

        self.upsample = nn.Upsample((H // 4, W // 4))

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x11 = self.avgpool1(x1)
        x22 = torch.cat([x11, x2], 1)
        x22 = self.avgpool2(x22)

        x4 = self.upsample(x4)
        x33 = torch.cat([x22, x3, x4], 1)

        return self.final(x33), x1, x2


class MappingNetwork(nn.Module):
    def __init__(self, output, H=256, W=256):
        super(MappingNetwork, self).__init__()

        self.layer1 = nn.Sequential(
            generate_group(256, 128, 3, 2),
            generate_group(128, 128, 3, 2)
        )

        self.layer2 = nn.Sequential(
            generate_group(192, 128, 3, 1),
            generate_group(128, 128, 3, 1)
        )

        self.layer3 = nn.Sequential(
            generate_group(160, 128, 3, 1),
            generate_group(128, 128, 3, 1)
        )

        self.final = nn.Conv2d(128, output, 3, 1)

    def forward(self, C, C1, C2):
        x1 = self.layer1(C)
        x2 = F.interpolate(x1, size=C2.shape[2:])
        x3 = torch.cat([x2, C2], 1)

        x4 = self.layer2(x3)
        x5 = F.interpolate(x4, size=C1.shape[2:])
        x6 = torch.cat([x5, C1], 1)

        x7 = self.layer3(x6)
        x8 = self.final(x7)

        return x8


class MapModel(nn.Module):
    def __init__(self, cin, cout, H=256, W=256):
        self.backbone = MapBackBone(cin, H, W)
        self.header = MappingNetwork(cout, H, W)

    def forward(self, x):
        x1, x2, x3 = self.backbone(x)
        return self.header(x1, x2, x3)


if __name__ == "__main__":
    x = torch.randn(1, 5, 256, 256)
    net = MapBackBone(5)
    net.train()

    x, x1, x2 = net(x)
    net2 = MappingNetwork(1)
    net2.train()
    x = net2(x, x1, x2)
    print(x.shape)
