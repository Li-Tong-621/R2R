import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os
import math
from utils import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import numpy as np
from cv2 import imread
from skimage import img_as_float
import time
import sys
def im2patch(im, pch_size, stride=1):
    '''
    Transform image to patches.
    Input:
        im: 3 x H x W or 1 X H x W image, numpy format
        pch_size: (int, int) tuple or integer
        stride: (int, int) tuple or integer
    '''
    if isinstance(pch_size, tuple):
        pch_H, pch_W = pch_size
    elif isinstance(pch_size, int):
        pch_H = pch_W = pch_size
    else:
        sys.exit('The input of pch_size must be a integer or a int tuple!')

    if isinstance(stride, tuple):
        stride_H, stride_W = stride
    elif isinstance(stride, int):
        stride_H = stride_W = stride
    else:
        sys.exit('The input of stride must be a integer or a int tuple!')


    C, H, W = im.shape
    num_H = len(range(0, H-pch_H+1, stride_H))
    num_W = len(range(0, W-pch_W+1, stride_W))
    num_pch = num_H * num_W
    pch = np.zeros((C, pch_H*pch_W, num_pch), dtype=float)
    kk = 0
    for ii in range(pch_H):
        for jj in range(pch_W):
            temp = im[:, ii:H-pch_H+ii+1:stride_H, jj:W-pch_W+jj+1:stride_W]
            pch[:, kk, :] = temp.reshape((C, num_pch))
            kk += 1

    return pch.reshape((C, pch_H, pch_W, num_pch))

def noise_estimate(im, pch_size=8):
    '''
    Implement of noise level estimation of the following paper:
    Chen G , Zhu F , Heng P A . An Efficient Statistical Method for Image Noise Level Estimation[C]// 2015 IEEE International Conference
    on Computer Vision (ICCV). IEEE Computer Society, 2015.
    Input:
        im: the noise image, H x W x 3 or H x W numpy tensor, range [0,1]
        pch_size: patch_size
    Output:
        noise_level: the estimated noise level
    '''

    if im.ndim == 3:
        im = torch.permute(im, (2, 0, 1))
    else:
        im = np.expand_dims(im, axis=0)

    # image to patch
    pch = im2patch(im, pch_size, 3)  # C x pch_size x pch_size x num_pch tensor
    num_pch = pch.shape[3]
    pch = pch.reshape((-1, num_pch))  # d x num_pch matrix
    d = pch.shape[0]

    mu = pch.mean(axis=1, keepdims=True)  # d x 1
    X = pch - mu
    sigma_X = np.matmul(X, X.transpose()) / num_pch
    sig_value, _ = np.linalg.eigh(sigma_X)
    sig_value.sort()

    for ii in range(-1, -d-1, -1):
        tau = np.mean(sig_value[:ii])
        if np.sum(sig_value[:ii]>tau) == np.sum(sig_value[:ii] < tau):
            return np.sqrt(tau)



from scipy.linalg import sqrtm

class R2RDenoisingDataset(Dataset):
    def __init__(self, y_tensor, alpha = 0.75):
        '''
        Inputs:
        - y_tensor: tensor of cropped images (shape N x H x W x 3)
        - alpha: parameter of recorruption used in paper (default 0.75)
        '''
        super(R2RDenoisingDataset, self).__init__()
        self.y_tensor = y_tensor
        N,H,W,_=y_tensor.shape
        self.Y1 = torch.zeros(N, 3, H, W)
        self.Y2 = torch.zeros(N, 3, H, W)
        for i in range(len(self.y_tensor)):
            z = torch.randn(H, W).double()
            image = y_tensor[i]
            A = image.clone()
            B = image.clone()
            sigma = noise_estimate(image, pch_size=32)
            for c in range(3):
                #print((20 * sigma * np.identity(512)).shape,z.shape)
                #print((torch.matmul(torch.from_numpy(20 * sigma * np.identity(512)), z)).shape)
                A[:, :, c] = A[:, :, c] + torch.matmul(torch.from_numpy(20 * sigma * np.identity(H)), z)
                B[:, :, c] = B[:, :, c] - torch.matmul(torch.from_numpy(sigma * np.identity(H) / 20), z)
            self.Y1[i] = A.permute(2, 0, 1) #y hat
            self.Y2[i] = B.permute(2, 0, 1) #y tilde

    def __len__(self):
        return len(self.y_tensor)

    def __getitem__(self, idx):
        return self.Y1[idx], self.Y2[idx]

import torch.nn.init as init
from torch.optim.lr_scheduler import MultiStepLR
from models import DnCNN
num_epochs = 8000
batch_size = 1
# model = DnCNN(channels=3, num_of_layers=17).to(device)
criterion = nn.MSELoss(reduction = 'sum')
init_learning_rate = 0.001
# optimizer = torch.optim.Adam(model.parameters(), init_learning_rate)
# lr_control = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)
import argparse
parser = argparse.ArgumentParser(description='choices')
parser.add_argument('--cropped_path', default='./data/CC/Noisy/', type=str)
parser.add_argument('--save_path', default='./results/CC/', type=str)
args = parser.parse_args()
cropped_path = args.cropped_path
save_checkpoint = os.path.join(args.save_path, 'model')
save_img1 = os.path.join(args.save_path, 'img1')
save_img2 = os.path.join(args.save_path, 'img2')
os.makedirs( save_checkpoint  ,exist_ok=True)
os.makedirs( save_img1  ,exist_ok=True)
os.makedirs( save_img2  ,exist_ok=True)

cropped_img_list = os.listdir(cropped_path)
N = len(cropped_img_list) #number of noisy and ground truth
img = cv2.imread(cropped_img_list[0])
H, W, _ = img.shape
# H, W = 512, 512
Cropped_tensor = torch.zeros(N, H, W, 3) #real_JPG are noisy
from natsort import natsorted

for i, img_name in enumerate(natsorted(cropped_img_list)):
    img_path = cropped_path + '/' + img_name
    img = cv2.imread(img_path) / 255.0
    Cropped_tensor[i] = torch.from_numpy(img)

train_dataset = R2RDenoisingDataset(Cropped_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)


train_mse = torch.zeros(num_epochs)
test_mse = torch.zeros(num_epochs)
for idx, (y1_batch, y2_batch) in enumerate(train_loader):
    #avg_train_mse = 0.0
    model = DnCNN(channels=3, num_of_layers=17).to(device)
    model.apply(weights_init_kaiming)
    optimizer = torch.optim.Adam(model.parameters(), init_learning_rate)
    lr_control = MultiStepLR(optimizer, milestones=[int(0.3*num_epochs),int(0.6*num_epochs),int(0.9*num_epochs)], gamma=0.2)
    model.train()
    for epoch in range(num_epochs):
        
        optimizer.zero_grad()
        model_input = Variable(y1_batch.to(device))
        expected_out = Variable(y2_batch.to(device))
        out = model(model_input)
        loss = criterion(out, expected_out)
        loss.backward()
        optimizer.step()

        batch_mse = criterion(out, expected_out)
        avg_train_mse = batch_mse.item() / batch_size
        if epoch%10==0:
            print('epoch:',epoch,'  loss:',avg_train_mse)
    torch.save(model.state_dict(), os.path.join(save_checkpoint, str(idx)+'checkpoint.pth'))
    model.eval()
    model_input = Cropped_tensor[idx].permute(2, 0, 1).to(device)
    with torch.no_grad():
        model_output = model(torch.unsqueeze(model_input, 0))
    model_output = torch.squeeze(model_output).permute(1, 2, 0).float().clamp_(0, 1).detach().cpu().numpy()
    model_output = np.uint8((model_output *255.0).round())
    cv2.imwrite(os.path.join(save_img1, str(idx)+'.png'), model_output)

    model_input = Variable(y1_batch.to(device))
    with torch.no_grad():
        model_output = model(model_input)
    model_output = torch.squeeze(model_output).permute(1, 2, 0).float().clamp_(0, 1).detach().cpu().numpy()
    model_output = np.uint8((model_output *255.0).round())
    cv2.imwrite(os.path.join(save_img2, str(idx)+'.png'), model_output)
    print(str(idx),' get!')

