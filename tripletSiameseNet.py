import torch.nn as nn
import torchvision.models as models
import torch


class tripletSiameseNet(nn.Module):
    def __init__(self):
        super(tripletSiameseNet, self).__init__()
        self.branch = models.alexnet()
        self.branch.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=9216, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True)
        )
        self.fc8 = nn.Linear(in_features=4096 * 3, out_features=2)

    def forward(self, x):
        out1 = self.branch(x[:, 0, :, :, :])
        out2 = self.branch(x[:, 1, :, :, :])
        out3 = self.branch(x[:, 2, :, :, :])
        out = self.fc8(torch.cat([out1, out2, out3], dim=1))
        return out


if __name__ == '__main__':
    net = tripletSiameseNet()
    x = torch.randn(size=(256, 3, 3, 224, 244))
    print(net)