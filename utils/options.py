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
        parser.add_argument("--lr", type=float, help="Learning Rate", default=1e-4)
        parser.add_argument("--num_workers", type=float, help="The number of loader workers", default=8)
        parser.add_argument("--milestone", type=list, help="Learning Rate Scheduler Milestones",
                            default=[x for x in range(50, 101, 10)])
        parser.add_argument("--gamma", type=float, help="Learning Rate Scheduler Gamma", default=0.5)
        parser.add_argument("--split_scale", type=float, help="Validation set proportion", default=0.01)
        parser.add_argument("--CropSize", type=int, help="Training image crop size", default=64)
        parser.add_argument("--batch_size", type=int, default=32)

        # Data Parameter
        parser.add_argument("--use_gpus", action='store_true', help="Usage of GPUs", default=True)

        parser.add_argument("--load_weight", action='store_true', help="Load model weight before training")
        parser.add_argument("--ckp", type=str, help="The path of model weight file",
                            default="pretrained/model_deblur_best.pth")

        parser.add_argument("--upsample_scale", type=int, help="Upsample image scale", default=1)
        parser.add_argument("--LogPath", type=str, help="The path of log info",
                            default="LogFiles")

        parser.add_argument("--TrainImgPath", type=str, help="The path of train blurred image")
        parser.add_argument("--TrainEvePath", type=str, help="The path of train event data")
        parser.add_argument("--TrainGTPath", type=str, help="The path of train sharp image")

        parser.add_argument("--TestImgPath", type=str, help="The path of test blurred image",
                            default="demo_input/lr")
        parser.add_argument("--TestEvePath", type=str, help="The path of test event data",
                            default="demo_input/events")
        parser.add_argument("--TestSavePath", type=str, help="The saving path of test result",
                            default="results")
        self.parser = parser

    def parse(self):
        return self.parser.parse_args()
