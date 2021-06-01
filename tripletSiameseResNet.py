import torch.nn as nn
import torchvision.models as models
import torch


class tripletSiameseResNet(nn.Module):
    def __init__(self):
        super(tripletSiameseResNet, self).__init__()
        self.branch = models.resnet18()
        self.addedClassifier = nn.Linear(in_features=1000 * 3, out_features=2, bias=True)

    def forward(self, x):
        out1 = self.branch(x[:, 0, :, :, :])
        out2 = self.branch(x[:, 1, :, :, :])
        out3 = self.branch(x[:, 2, :, :, :])
        out = self.addedClassifier(torch.cat([out1, out2, out3], dim=1))
        return out


if __name__ == '__main__':
    net = tripletSiameseResNet()
    print(net)
    x = torch.randn(size=(32, 3, 3, 224, 244))
    print(net(x).shape)