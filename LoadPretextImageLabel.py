import torchvision.transforms as transform
import torchvision.transforms._transforms_video as v_transform
import torchvision.io as io
from torch.utils.data import Dataset, DataLoader
import config



class FrameOrderVerificationData(Dataset):
    # read all tuple frame's filename and label
    def __init__(self, UCF_root, transform=None):
        # read tuple frame index
        self.tupleData = []
        self.transform = transform
        self.UCF_root = UCF_root

        with open('train01_image_keys/train01_image_keys.txt', 'r') as f:
            all_lines = f.readlines()
            for line in all_lines:
                tupleFrames = line.split('\t')
                tupleFrames = tupleFrames[:-1]

                video_name = tupleFrames[0].split('/')[0]
                action_name = video_name.split('_')[1]

                for i in range(len(tupleFrames)):
                    tupleFrames[i] = int(tupleFrames[i].split('_')[-1])

                self.tupleData.append([video_name, action_name, tupleFrames])

        # read tuple frame label
        with open('train01_image_keys/train01_image_labs.txt', 'r') as f:
            all_lines = f.readlines()
            for i in range(len(all_lines)):
                label = all_lines[i].split('\n')[0]
                self.tupleData[i].append(int(label))

        self.len = len(self.tupleData)


    # Get a tuple frames and label
    def __getitem__(self, index):
        video_path = self.UCF_root + self.tupleData[index][1] + '/' + self.tupleData[index][0] + '.avi'
        #
        # print(video_path)
        # print(self.tupleData[index])

        vframes, _, _ = io.read_video(video_path)

        tupleFrames = vframes[self.tupleData[index][2]]

        label = self.tupleData[index][-1]


        # May use transform function to transform samples
        if self.transform is not None:
            tupleFrames = self.transform(tupleFrames)

        return tupleFrames, label


    # get the length of dataset
    def __len__(self):
        return self.len



# define the transformation
transform = transform.Compose([
    v_transform.ToTensorVideo(),
    v_transform.RandomHorizontalFlipVideo(),
    v_transform.RandomResizedCropVideo(224),
])


# load the UCF101 training dataset
trainset = FrameOrderVerificationData(
    UCF_root=config.UCF101_Dataset_root,
    transform=transform
)

# divide the dataset into batches
trainset_loader = DataLoader(
    trainset,
    batch_size=config.SSL_batch_size,
    shuffle=True,
    num_workers=8
)
