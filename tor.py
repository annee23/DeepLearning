from scipy._lib.decorator import __init__
from torch.distributions import transforms
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
from PIL import Image
import sift
import time
start = time.time()
ima_dir = "./IMG"
data_path = ima_dir + "/*.JPG"
files = glob.glob(data_path)
data = []
comp = cv2.imread("DSC01873.JPG", 0)
comp = cv2.resize(comp, dsize=(256, 256), interpolation=cv2.INTER_AREA)
print(comp)
kp, des, patches = sift.computeKeypointsAndDescriptors(comp)
cnt_file = []
cnt = []
for i in range(len(kp)):
    cnt_file.append(0)
    cnt.append(0)
sift_cv2 = cv2.xfeatures2d.SIFT_create()
labels = []
for k in range(len(patches)):
    patches[k] = np.array(patches[k])
    # arr = patches[k].reshape(50,50)
    # im = Image.fromarray(arr.astype('uint8'),'L')
    print(type(patches[k]))
    if patches[k].shape[0]!= 0 and patches[k].shape[1]!=0:
        im = Image.fromarray(patches[k], 'L')
        im.save('myim.png')
        im = cv2.imread('myim.png')
        kp, des = sift_cv2.detectAndCompute(im, None)
    for fl in files:
        img = cv2.imread(fl, 0)
        img = cv2.resize(img, dsize=(256,256), interpolation=cv2.INTER_AREA)
        kp1, des1 = sift_cv2.detectAndCompute(img,None)
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des, des1, k=2)

        # Need to draw only good matches, so create a mask
        matchesMask = [[0, 0] for i in range(len(matches))]
        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.3 * n.distance:
                matchesMask[i] = [1, 0]
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=0)
        img3 = cv2.drawMatchesKnn(im, kp, img, kp1, matches, None, **draw_params)
        plt.imshow(img3, ), plt.show()
        cnt[k] = len(matches)
    if cnt[k] > 0:
        cnt_file[k] = cnt[k]
    labels.append(cnt_file[k])
    print(cnt_file[k])
    print("time :", time.time() - start)


class patDataset(Dataset):
    def __init__(self, patches, labels):
        self.labels = labels
        self.patches = patches

    def __len__(self, patches):
        return len(self.patches)

    def __getitem__(self, idx):
        X = self.patches[idx]
        y = self.labels[idx]

        return X, y

from torchvision import transforms, utils
from torch.utils.data.dataset import random_split
import torchvision
trans = transforms.Compose([transforms.Resize(32, 32), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.456),(0.229,0.224, 0.225))])
train_ = patDataset.__len__(patches)*0.8
test_ = patDataset.__len__(patches)*0.2
train_dataset, val_dataset = random_split(patDataset(Dataset), [train_, test_])
train_loader = DataLoader(dataset=train_dataset, batch_size=16)
val_loader = DataLoader(dataset=val_dataset, batch_size=20)

use_cuda = torch.cuda.is_available()
import itertools
from IPython.display import Image
from IPython import display
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class CNNClassifier(nn.Module):

    def __init__(self):
        # 항상 torch.nn.Module을 상속받고 시작
        super(CNNClassifier, self).__init__()
        conv1 = nn.Conv2d(1, 6, 5, 1)  # 6@24*24
        # activation ReLU
        pool1 = nn.MaxPool2d(2)  # 6@12*12
        conv2 = nn.Conv2d(6, 16, 5, 1)  # 16@8*8
        # activation ReLU
        pool2 = nn.MaxPool2d(2)  # 16@4*4

        self.conv_module = nn.Sequential(
            conv1,
            nn.ReLU(),
            pool1,
            conv2,
            nn.ReLU(),
            pool2
        )

        fc1 = nn.Linear(16 * 4 * 4, 120)
        # activation ReLU
        fc2 = nn.Linear(120, 84)
        # activation ReLU
        fc3 = nn.Linear(84, 10)

        self.fc_module = nn.Sequential(
            fc1,
            nn.ReLU(),
            fc2,
            nn.ReLU(),
            fc3
        )

        # gpu로 할당
        if use_cuda:
            self.conv_module = self.conv_module.cuda()
            self.fc_module = self.fc_module.cuda()

    def forward(self, x):
        out = self.conv_module(x)  # @16*4*4
        # make linear
        dim = 1
        for d in out.size()[1:]:  # 16, 4, 4
            dim = dim * d
        out = out.view(-1, dim)
        out = self.fc_module(out)
        return F.softmax(out, dim=1)

cnn = CNNClassifier()
# loss
criterion = nn.CrossEntropyLoss()
# backpropagation method
learning_rate = 1e-3
optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
# hyper-parameters
num_epochs = 2
num_batches = len(train_loader)

trn_loss_list = []
val_loss_list = []
for epoch in range(num_epochs):
    trn_loss = 0.0
    for i, data in enumerate(train_loader):
        x, label = data
        if use_cuda:
            x = x.cuda()
            label = label.cuda()
        # grad init
        optimizer.zero_grad()
        # forward propagation
        model_output = cnn(x)
        # calculate loss
        loss = criterion(model_output, label)
        # back propagation
        loss.backward()
        # weight update
        optimizer.step()

        # trn_loss summary
        trn_loss += loss.item()
        # del (memory issue)
        del loss
        del model_output

        # 학습과정 출력
        if (i + 1) % 100 == 0:  # every 100 mini-batches
            with torch.no_grad():  # very very very very important!!!
                val_loss = 0.0
                for j, val in enumerate(val_loader):
                    val_x, val_label = val
                    if use_cuda:
                        val_x = val_x.cuda()
                        val_label = val_label.cuda()
                    val_output = cnn(val_x)
                    v_loss = criterion(val_output, val_label)
                    val_loss += v_loss

            print("epoch: {}/{} | step: {}/{} | trn loss: {:.4f} | val loss: {:.4f}".format(
                epoch + 1, num_epochs, i + 1, num_batches, trn_loss / 100, val_loss / len(val_loader)
            ))

            trn_loss_list.append(trn_loss / 100)
            val_loss_list.append(val_loss / len(val_loader))
            trn_loss = 0.0
