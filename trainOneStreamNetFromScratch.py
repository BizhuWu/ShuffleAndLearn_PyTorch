from LoadUCF101Data import trainset_loader, testset_loader
from onlyOneBranchOfTwoStreamNet import oneStreamNet
import torch
import torch.optim as optim
import torch.nn.functional as F
import config



if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')



net = oneStreamNet().to(device)



optimizer = optim.SGD(
    params=net.parameters(),
    lr=config.AR_lr
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
            RGBFrame, label = data
            RGBFrame = RGBFrame.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            output = net(RGBFrame)
            loss = F.cross_entropy(output, label)

            loss.backward()
            optimizer.step()

            print("Epoch " + str(i+1) + ", Iteration " + str(index+1) + "'s Loss: " + str(loss.item()))
            with open('log_TrainOnUCFFromScratch.txt', 'a') as f:
                f.write("Epoch " + str(i+1) + ", Iteration " + str(index+1) + "'s Loss: " + str(loss.item()) + "\n")

        save_checkpoint('model_TrainOnUCFFromScratch/checkpoint-epoch-%i.pth' % (i+1), net, optimizer)
        test(i+1)



def test(i_epoch):

    net.eval()

    correct = 0

    with torch.no_grad():
        for index, data in enumerate(testset_loader):
            RGBFrame, label = data

            RGBFrame = RGBFrame.to(device)
            label = label.to(device)

            output = net(RGBFrame)

            max_value, max_index = output.max(1, keepdim=True)
            correct += max_index.eq(label.view_as(max_index)).sum().item()

    print("Accuracy: " + str(correct*1.0*100/len(testset_loader.dataset)))
    with open('log_TrainOnUCFFromScratch.txt', 'a') as f:
        f.write("Epoch " + str(i_epoch) + "'s Accuracy: " + str(correct*1.0*100/len(testset_loader.dataset)) + "\n")



if __name__ == '__main__':
    train(config.AR_epoch)