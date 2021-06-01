import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
from PIL import Image
import torchvision.transforms._transforms_video as v_transform
import torchvision.io as io
import config



classInd = []
with open('classInd.txt', 'r') as f:
    all_Class_and_Ind = f.readlines()
    for line in all_Class_and_Ind:
        idx = line[:-1].split(' ')[0]
        className = line[:-1].split(' ')[1]
        classInd.append(className)

TrainVideoNameList = []
with open('trainlist01.txt', 'r') as f:
    all_Class_and_Ind = f.readlines()
    for line in all_Class_and_Ind:
        video_name = line.split(' ')[0]
        video_name = video_name.split('/')[1]
        TrainVideoNameList.append(video_name)

TestVideoNameList = []
with open('testlist01.txt', 'r') as f:
    all_Class_and_Ind = f.readlines()
    for line in all_Class_and_Ind:
        video_name = line[:-1].split(' ')[0]
        video_name = video_name.split('/')[1]
        TestVideoNameList.append(video_name)



class UCF101Data(Dataset):  # define a class named MNIST
    # read all pictures' filename
    def __init__(self, UCF_root, isTrain, transform=None):
        # root: Dataset's filepath
        # classInd: dictionary (1 -> ApplyEyeMakeup)
        self.RGBvideoData = []
        self.transform = transform

        for i in range(0, 101):
            RGB_class_path = UCF_root + classInd[i]

            # only load train/test data using TrainVideoNameList/TestVideoNameList
            if isTrain:
                TrainOrTest_VideoNameList = list(set(os.listdir(RGB_class_path)).intersection(set(TrainVideoNameList)))
            else:
                TrainOrTest_VideoNameList = list(set(os.listdir(RGB_class_path)).intersection(set(TestVideoNameList)))


            for video_dir in os.listdir(RGB_class_path):
                if video_dir in TrainOrTest_VideoNameList:
                    signel_RGB_video_path = RGB_class_path + '/' + video_dir

                    # (signel_RGB_video_path, label)
                    self.RGBvideoData.append((signel_RGB_video_path, i))

        self.len = len(self.RGBvideoData)


    # Get a sample from the dataset & Return an image and it's label
    def __getitem__(self, index):
        signel_RGB_video_path, label = self.RGBvideoData[index]

        vframes, _, _ = io.read_video(signel_RGB_video_path)

        ran_i = np.random.randint(len(vframes))

        # open the RGB frame
        RGB_frame = vframes[ran_i]
        RGB_frame = RGB_frame.numpy()

        # May use transform function to transform samples
        if self.transform is not None:
            RGB_frame = self.transform(RGB_frame)

        return RGB_frame, label


    # get the length of dataset
    def __len__(self):
        return self.len



# define the transformation
# PIL images -> torch tensors [0, 1]
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
])


# load the UCF101 training dataset
trainset = UCF101Data(
    UCF_root=config.UCF101_original_Dataset_root,
    isTrain=True,
    transform=transform
)

# divide the dataset into batches
trainset_loader = DataLoader(
    trainset,
    batch_size=config.AR_TRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=8
)



# load the UCF101 testing dataset
testset = UCF101Data(
    UCF_root=config.UCF101_original_Dataset_root,
    isTrain=False,
    transform=transform
)

# divide the dataset into batches
testset_loader = DataLoader(
    testset,
    batch_size=config.AR_TEST_BATCH_SIZE,
    shuffle=False,
    num_workers=8
)