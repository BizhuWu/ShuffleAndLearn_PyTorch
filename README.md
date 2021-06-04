# ShuffleAndLearn_PyTorch

This repository is an **`PyTorch`** implementation of the paper ["Shuffle and Learn: Unsupervised Learning using Temporal Order Verification"](https://arxiv.org/pdf/1603.08561.pdf). 

Note that I used **`ResNet-18`** instead of **`AlexNet`** as my backbone.

The original repository of the paper is [imisra/shuffle-tuple](https://github.com/imisra/shuffle-tuple) (Caffe Version)

The repository includes the whole training process. 

Specifically, I use PyTorch 1.7 **`VideoIO / Video Datasets Loading API / Video Transform`** to process the Data. [More Details：How to use Video Datasets，Video IO，Video Classification Models，Video Transform in PyTorch](https://blog.csdn.net/qq_36627158/article/details/113791050)


&nbsp;

## Training Environment
+ Ubuntu 16.04.7 LTS
+ CUDA Version: 10.1
+ PyTorch 1.7.1
+ torchvision 0.8.2
+ numpy 1.19.2
+ pillow 8.1.0
+ python 3.8.5
+ av 8.0.3
+ matplotlib 3.3.4

&nbsp;

## Data Preparation
1. The tuples used for training our model can be downloaded as a zipped text file [here](https://onedrive.live.com/?cid=ad2f6792017eca5b&id=AD2F6792017ECA5B%214906&authkey=!AN5DFQ2InIXW7j4). 

Unzip it, and you will get two files. Then put it into the root directory.

```
Project
│--- train01_image_keys
│------ train01_image_keys.txt
│------ train01_image_labs.txt
│--- other files
```

Each line of the file **`train01_image_keys.txt`** defines a tuple of three frames. 

The corresponding file **`train01_image_labs.txt`** has a binary label indicating whether the tuple is in the correct or incorrect order.```

&nbsp;

2. UCF101 Dataset：[UCF101](https://www.crcv.ucf.edu/data/UCF101.php)

After downloading the UCF101 dataset: **`UCF101.rar`**, you should unrar it. Then put it into the directory named **`data`**
```
Project
│--- data
│------ UCF101
│--------- UCF-101
│------------ ApplyEyeMakeup
│------------ ApplyLipstick
│------------ ...
│--- other files
```

**Note**: The **`HandstandPushups`** class's name on UCF-101 is inconsistent with the videos' name inside. 

The **`s`** in directory's name is lower-case (Hand**s**tandPushups) while the **`s`** in the videos' name inside is upper-case (Hand**S**tandPushups).

```
HandstandPushups
│--- v_HandStandPushups_g01_c01.avi
│--- ...
```

So, I copied the orginal UCF dataset, and then the copied one as **`UCF-101-original`**. Next, I renamed the **`HandstandPushups`** class on the orginal UCF dataset **`UCF-101`** name as **`HandStandPushups`**.

Now, the dataset's structure looks like this:
```
Project
│--- data
│------ UCF101
│--------- UCF-101-original
│------------ ApplyEyeMakeup
│------------ ApplyLipstick
│------------ ...
│------------ HandstandPushups
│------------ ...
│--------- UCF-101
│------------ ApplyEyeMakeup
│------------ ApplyLipstick
│------------ ...
│------------ HandStandPushups
│------------ ...
│--- other files
```

&nbsp;

## Train
### SSL pretext Task Training
Before training, make sure you have a directory named **`model_SSL`** in the root project to save checkpoint file.
```python
python3 frameOrderVerificationTraining.py
```
### Action Recognition Training from Scratch
Before training, make sure you have a directory named **`model_TrainOnUCFFromSSL`** in the root project to save checkpoint file.
```python
python3 trainOneStreamNetFromSSL.py
```
### Action Recognition Training from SSL
Before training, make sure you have a directory named **`model_TrainOnUCFFromScratch`** in the root project to save checkpoint file.
```python
python3 trainOneStreamNetFromScratch.py
```

&nbsp;

## Performance
### 10 epoch
No.|Acc|Loss
:---:|:---:|:---:
1|![](/result_png/acc10_1.png)|![](/result_png/loss10_1.png)
2|![](/result_png/acc10_2.png)|![](/result_png/loss10_2.png)
3|![](/result_png/acc10_3.png)|![](/result_png/loss10_3.png)

### 50 epoch
No.|Acc|Loss
:---:|:---:|:---:


