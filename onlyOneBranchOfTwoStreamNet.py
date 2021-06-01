import torch.nn as nn
import torchvision.models as models
import torch


class oneStreamNet(nn.Module):
    def __init__(self):
        super(oneStreamNet, self).__init__()
        # self.branch = models.alexnet()
        # # fc8
        # self.branch.classifier[6] = nn.Linear(in_features=4096, out_features=101, bias=True)

        self.branch = models.resnet18()
        self.branch.fc = nn.Linear(in_features=512, out_features=101, bias=True)


    def forward(self, x):
        out = self.branch(x)
        return out


if __name__ == '__main__':
    net = oneStreamNet()
    print(net)
    x = torch.randn(size=(256, 3, 224, 244))
    print(net(x).shape)