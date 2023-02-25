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

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x11 = self.avgpool1(x1)
        x22 = torch.cat([x11, x2], 1)
        x22 = self.avgpool2(x22)

        x44 = F.interpolate(x4, size=x22.shape[2:])
        x33 = torch.cat([x22, x3, x44], 1)

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
        self.H = H
        self.W = W

    def forward(self, C, C1, C2):
        x1 = self.layer1(C)
        x2 = F.interpolate(x1, size=C2.shape[2:])
        x3 = torch.cat([x2, C2], 1)

        x4 = self.layer2(x3)
        x5 = F.interpolate(x4, size=C1.shape[2:])
        x6 = torch.cat([x5, C1], 1)

        x7 = self.layer3(x6)
        x8 = self.final(x7)

        return torch.sigmoid(F.interpolate(x8, size=(self.W, self.H)))


class LocalizationHeaderNetwork(nn.Module):
    def __init__(self, output):
        super(LocalizationHeaderNetwork, self).__init__()
        self.fc = None
        self.output = output

    def forward(self, x):
        x = x.view(x.size(0), -1)
        if self.fc is None:
            self.fc = nn.Linear(x.size(1), 2)
        x = self.fc(F.relu(x))
        return x


class LocalizationModel(nn.Module):
    def __init__(self):
        super(LocalizationModel, self).__init__()
        self.backbone = MapBackBone(2)
        self.head = LocalizationHeaderNetwork(2)

    def forward(self, x):
        x, _, _ = self.backbone(x)
        return self.head(x)


class MapModel(nn.Module):
    def __init__(self, cin, cout, H=256, W=256):
        super(MapModel, self).__init__()
        self.backbone = MapBackBone(cin, H, W)
        self.header = MappingNetwork(cout, H, W)

    def forward(self, x):
        x1, x2, x3 = self.backbone(x)
        return self.header(x1, x2, x3)


if __name__ == "__main__":
    x = torch.randn(8, 2, 256, 256)
    net = LocalizationModel()
    net.train()

    x = net(x)
    print(x.shape)
