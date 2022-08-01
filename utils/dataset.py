import os
import os.path
import numpy as np
import random
import torch
import cv2
import glob
import torch.utils.data as udata


def read_img(img_path):
    img = cv2.imread(img_path)
    y, cr, cb = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb))
    return y,cr,cb

class Dataset_Train(udata.Dataset):
    def __init__(self, ImgPath, EvePath, GTPath, CropSize, mode):
        super(Dataset_Train, self).__init__()
        self.imgList = sorted(glob.glob(os.path.join(ImgPath, '*.png')))
        self.eveList = sorted(glob.glob(os.path.join(EvePath, '*.npy')))
        self.gtList = sorted(glob.glob(os.path.join(GTPath, '*.png')))

        if len(self.imgList) != len(self.eveList) or len(self.imgList) != len(self.gtList):
            raise ValueError("Data is unpaired!")

        self.CropSize = CropSize
        self.mode = mode

    def __len__(self):
        return len(self.imgList)

    @staticmethod
    def MyRandomCrop(blur, sharp, events, CropSize):
        (_, h, w) = events.shape
        i = random.randint(0, h - CropSize - 1)
        j = random.randint(0, w - CropSize - 1)

        blur_patch = blur[i * 4:(i + CropSize) * 4, j * 4:(j + CropSize) * 4]
        sharp_patch = sharp[i * 4:(i + CropSize) * 4, j * 4:(j + CropSize) * 4]
        event_patch = events[:, i:i + CropSize, j:j + CropSize]
        return blur_patch, sharp_patch, event_patch

    @staticmethod
    def MyRandomCrop2(blur, sharp, events, CropSize):
        (h,w) = blur.shape
        i = random.randint(0, h - CropSize - 1)
        j = random.randint(0, w - CropSize - 1)

        blur_patch = blur[i:i + CropSize, j:j + CropSize]
        sharp_patch = sharp[i:i + CropSize, j:j + CropSize]
        event_patch = events[:, i:i + CropSize, j:j + CropSize]
        return blur_patch, sharp_patch, event_patch

    def __getitem__(self, index):
        imgs = self.imgList[index]
        gts = self.gtList[index]
        ev = self.eveList[index]

        imgs, _, _ = read_img(imgs)
        gts, _, _ = read_img(gts)
        ev = np.load(ev, allow_pickle=True)
        
        if self.mode == 'sr':
            imgs, gts, ev = self.MyRandomCrop(imgs, gts, ev, self.CropSize)
        else:
            imgs, gts, ev = self.MyRandomCrop2(imgs, gts, ev, self.CropSize)
        return torch.Tensor(np.expand_dims(imgs, 0)), torch.Tensor(ev), torch.Tensor(np.expand_dims(gts, 0))

class Dataset_Test(udata.Dataset):
    def __init__(self, ImgPath, EvePath):
        super(Dataset_Test, self).__init__()
        self.imgList = sorted(glob.glob(os.path.join(ImgPath, '*.png')))
        self.eveList = sorted(glob.glob(os.path.join(EvePath, '*.npy')))
        print(len(self.imgList),len(self.eveList))
        if len(self.imgList) != len(self.eveList):
            raise ValueError("Data is unpaired!")

    def __len__(self):
        return len(self.imgList)

    def __getitem__(self, index):
        blurimg = self.imgList[index]
        eveimg = self.eveList[index]
        _, name = os.path.split(blurimg)

        blur = read_img(blurimg)

        events = np.float32(np.load(eveimg, allow_pickle=True))
        return torch.Tensor(blur), torch.Tensor(events), name