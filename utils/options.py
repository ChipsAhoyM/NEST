"""
    Parse input arguments
"""

import argparse


class Options:
    def __init__(self):
        # Parse options for processing
        parser = argparse.ArgumentParser(description='D-Net/S-Net for Image Deblurring/SR with Event')

        parser.add_argument("--mode",type=str, help="Deblur or SR", default='deblur')
        
        # Training Parameter
        parser.add_argument("--train", action='store_true', help="train or test", default=False)
        parser.add_argument("--BegEpoch", type=int, help="The Begin Epoch", default=1)
        parser.add_argument("--NumEpoch", type=int, help="The Number of Epoch", default=100)
        parser.add_argument("--lr", type=float, help="Learning Rate", default=1e-3)
        parser.add_argument("--num_workers", type=float, help="The number of loader workers", default=8)
        parser.add_argument("--milestone", type=list, help="Learning Rate Scheduler Milestones",
                            default=[x for x in range(50, 101, 10)])
        parser.add_argument("--gamma", type=float, help="Learning Rate Scheduler Gamma", default=0.1)
        parser.add_argument("--split_scale", type=float, help="Validation set proportion", default=0.05)
        parser.add_argument("--CropSize", type=int, help="Training image crop size", default=128)
        parser.add_argument("--batch_size", type=int, default=8)

        # Data Parameter
        parser.add_argument("--use_gpus", action='store_true', help="Usage of GPUs", default=True)

        parser.add_argument("--load_weight", action='store_true', help="Load model weight before training")
        parser.add_argument("--ckp", type=str, help="The path ßof model weight file",
                            default="pretrained/model_best.pth")

        parser.add_argument("--LogPath", type=str, help="The path of log info",
                            default="LogFiles")

        parser.add_argument("--TrainImgPath", type=str, help="The path of train blurred image",
                            default="/data/eSL-Net/GrayScale/train_ave_bicubic")
        parser.add_argument("--TrainEvePath", type=str, help="The path of train event data",
                            default="/data/eSL-Net/Event/train_esim/divide")
        parser.add_argument("--TrainGTPath", type=str, help="The path of train sharp image",
                            default="/data/eSL-Net/GrayScale/train_sharp_gray_lr")

        parser.add_argument("--TestImgPath", type=str, help="The path of test blurred image",
                            default="/data/val/rgb-blur")
        parser.add_argument("--TestEvePath", type=str, help="The path of test event data",
                            default="/data/eSL-Net/Event/val_esim/divide")
        parser.add_argument("--TestSavePath", type=str, help="The saving path of test result",
                            default="results")
        self.parser = parser

    def parse(self):
        return self.parser.parse_args()
