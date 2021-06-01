from LoadPretextImageLabel import trainset_loader
from tripletSiameseResNet import tripletSiameseResNet
import torch
import torch.optim as optim
import torch.nn.functional as F
import config



if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')



net = tripletSiameseResNet().to(device)



optimizer = optim.SGD(
    params=net.parameters(),
    lr=config.SSL_lr
)


def save_checkpoint(path, model, optimizer):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, path)


def train(epoch):
    for i in range(epoch):
        net.train()

        for index, data in enumerate(trainset_loader):
            tripleFrames, label = data
            tripleFrames = tripleFrames.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            output = net(tripleFrames)
            loss = F.cross_entropy(output, label)

            loss.backward()
            optimizer.step()

            print("Epoch " + str(i+1) + ", Iteration " + str(index+1) + "'s Loss: " + str(loss.item()))
            with open('log_SSL.txt', 'a') as f:
                f.write("Epoch " + str(i+1) + ", Iteration " + str(index+1) + "'s Loss: " + str(loss.item()) + "\n")

        save_checkpoint('model_SSL/checkpoint-%i.pth' % i, net, optimizer)



if __name__ == '__main__':
    train(config.SSL_epoch)