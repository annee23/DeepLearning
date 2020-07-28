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
comp = cv2.resize(comp, dsize=(128, 128), interpolation=cv2.INTER_AREA)
print(comp)
kp, des, patches = sift.computeKeypointsAndDescriptors(comp)
cnt_file = []
cnt = []

for i in range(len(kp)):
    cnt_file.append(0)
    cnt.append(0)
patches_img = []
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
        #im = im.astype('float32')
        patches_img.append(im.astype('float32'))

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


# plt.hist(cnt_file)
# plt.show()

# labels = []
train_img = []
# for i in range(len(patches)):
#     labels.append(1)
for k in range(len(patches_img)):
    train_img.append(patches_img[k])
class patDataset(Dataset):
    def __init__(self, patches):
        self.patches = patches
        self.data_len = len(patches)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        X = self.patches[idx]


        return X
class labDataset(Dataset):
    def __init__(self, labels):
        self.labels = labels
        self.data_len = len(labels)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        X = self.labels[idx]


        return X

from torchvision import transforms, utils
from torch.utils.data.dataset import random_split
import torchvision

patDataset.__init__(self=patDataset, patches=patches)
trans = transforms.Compose([transforms.Resize(32, 32), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.456),(0.229,0.224, 0.225))])
train_ = int(patDataset.__len__(self=patDataset)*0.8)
test_ = patDataset.__len__(self=patDataset)-train_
# train_x, val_x = random_split(patDataset( patches=patches), [train_, test_])
# train_y, val_y = random_split(labDataset( labels=labels), [train_, test_])

# train_loader = DataLoader(dataset=train_dataset, batch_size=16)
#
# val_loader = DataLoader(dataset=val_dataset, batch_size=20)

use_cuda = torch.cuda.is_available()


# importing the libraries
import pandas as pd
import numpy as np

# for reading and displaying images
from skimage.io import imread
import matplotlib.pyplot as plt


# for creating validation set
from sklearn.model_selection import train_test_split

# for evaluating the model
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

# converting the list to numpy array
train_x = train_img
print(len(train_img))
# defining the target
train_y = np.array(labels)

# create validation set
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.2)
print((train_x.shape, train_y.shape), (val_x.shape, val_y.shape))

# converting training images into torch format
train_x = train_x.reshape(218, 1, 28, 28)
train_x  = torch.from_numpy(train_x)


# converting the target into torch format
train_y = train_y.astype(int);
train_y = torch.from_numpy(train_y)

# shape of training data
train_x.shape, train_y.shape

#converting validation images into torch format
val_x = val_x.reshape(25, 1, 28, 28)
val_x  = torch.from_numpy(val_x)

# converting the target into torch format
val_y = val_y.astype(int);
val_y = torch.from_numpy(val_y)

# shape of validation data
val_x.shape, val_y.shape
class Net(Module):
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(4 * 7 * 7, 10)
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


# defining the model
model = Net()
# defining the optimizer
optimizer = Adam(model.parameters(), lr=0.07)
# defining the loss function
criterion = CrossEntropyLoss()
# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

print(model)


def train(epoch):
    model.train()
    tr_loss = 0
    # getting the training set
    x_train, y_train = Variable(train_x), Variable(train_y)
    # getting the validation set
    x_val, y_val = Variable(val_x), Variable(val_y)
    # converting the data into GPU format
    if torch.cuda.is_available():
        x_train = x_train.cuda()
        y_train = y_train.cuda()
        x_val = x_val.cuda()
        y_val = y_val.cuda()

    # clearing the Gradients of the model parameters
    optimizer.zero_grad()

    # prediction for training and validation set
    output_train = model(x_train)
    output_val = model(x_val)

    # computing the training and validation loss
    loss_train = criterion(output_train, y_train)
    loss_val = criterion(output_val, y_val)
    train_losses.append(loss_train)
    val_losses.append(loss_val)

    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()
    if epoch % 2 == 0:
        # printing the validation loss
        print('Epoch : ', epoch + 1, '\t', 'loss :', loss_val)

# defining the number of epochs
n_epochs = 25
# empty list to store training losses
train_losses = []
# empty list to store validation losses
val_losses = []
# training the model
for epoch in range(n_epochs):
    train(epoch)
# plotting the training and validation loss
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
plt.show()

# prediction for training set
with torch.no_grad():
    output = model(train_x.cuda())

softmax = torch.exp(output).cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)

# accuracy on training set
accuracy_score(train_y, predictions)

# prediction for validation set
with torch.no_grad():
    output = model(val_x.cuda())

softmax = torch.exp(output).cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)

# accuracy on validation set
accuracy_score(val_y, predictions)

