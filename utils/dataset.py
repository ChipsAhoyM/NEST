"""
Code for reading the dataset
"""

import os
import os.path
import numpy as np
import random
import torch
import cv2
import glob
import torch.utils.data as udata


"""
Dataset for training
    ImgPath: path of input images
    EvePath: path of events stack (N*H*W)
    GTPath: path of ground truth images
    CropSize: crop size
    scale: scale factor
"""
class Dataset_Train(udata.Dataset):
    def __init__(self, ImgPath, EvePath, GTPath, CropSize, scale = 1):
        super(Dataset_Train, self).__init__()
        self.imgList = sorted(glob.glob(os.path.join(ImgPath, '*.png')))
        self.eveList = sorted(glob.glob(os.path.join(EvePath, '*.npy')))
        self.gtList = sorted(glob.glob(os.path.join(GTPath, '*.png')))

        if len(self.imgList) != len(self.eveList) or len(self.imgList) != len(self.gtList):
            raise ValueError("Data is unpaired!")

        self.CropSize = CropSize
        self.scale = scale

    def __len__(self):
        return len(self.imgList)

    def MyRandomCrop(self, input, gt, events):
        """
        Random crop for training
        """
        (_, h, w) = events.shape
        i = random.randint(0, h - self.CropSize - 1)
        j = random.randint(0, w - self.CropSize - 1)

        input_patch = input[i:i+self.CropSize, j:j+self.CropSize]
        gt_patch = gt[i * self.scale:(i + self.CropSize) * self.scale, j * self.scale:(j + self.CropSize) * self.scale]
        event_patch = events[:, i:i + self.CropSize, j:j + self.CropSize]
        return input_patch, gt_patch, event_patch

    def __getitem__(self, index):
        imgs = self.imgList[index]
        gts = self.gtList[index]
        ev = self.eveList[index]

        imgs = cv2.imread(imgs, cv2.IMREAD_GRAYSCALE)
        gts = cv2.imread(gts, cv2.IMREAD_GRAYSCALE)
        ev = np.load(ev, allow_pickle=True)
        
        imgs, gts, ev = self.MyRandomCrop(imgs, gts, ev)

        input = torch.Tensor(np.expand_dims(imgs,axis=0))
        gt = torch.Tensor(np.expand_dims(gts,axis=0))
        ev = torch.Tensor(ev)
        return {'input':input, 'ev':ev, 'gt':gt}

"""
Read the test images and transform to YCrCb color space
"""
def read_img(img_path):
    img = cv2.imread(img_path)
    y, cr, cb = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb))
    return  np.array([y,cr,cb])


"""
Dataset for testing
    ImgPath: path of input images
    EvePath: path of events stack (N*H*W)
"""
class Dataset_Test(udata.Dataset):
    def __init__(self, ImgPath, EvePath, scale = 1):
        super(Dataset_Test, self).__init__()
        self.imgList = sorted(glob.glob(os.path.join(ImgPath, '*.png')))
        self.eveList = sorted(glob.glob(os.path.join(EvePath, '*.npy')))
        self.scale = scale

        if len(self.imgList) != len(self.eveList):
            raise ValueError("Data is unpaired!")

    def __len__(self):
        return len(self.imgList)

    def __getitem__(self, index):
        imgs = self.imgList[index]
        ev = self.eveList[index]
        _, name = os.path.split(imgs)

        input = read_img(imgs)
        upsample_input = cv2.resize(input.transpose(1,2,0), (input.shape[2] * self.scale, input.shape[1] * self.scale), interpolation=cv2.INTER_CUBIC).transpose(2,0,1)
        events = np.float32(np.load(ev, allow_pickle=True))
        return {'input':torch.Tensor(input), 'ev':torch.Tensor(events), 'upsample':upsample_input, 'name':name[:-4]}