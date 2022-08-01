import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import DeblurNet, SRNet
from utils import Options, Dataset_Test

args = Options().parse()
if args.use_gpus:
    device = torch.device("cuda")
    device_ids = [Id for Id in range(torch.cuda.device_count())]
else:
    device = torch.device("cpu")


def denorm(x):
    return (x + 1.0) * 255.0 / 2.0

def test():
    os.makedirs(args.TestSavePath, exist_ok=True)
    if args.mode == 'sr':
        model = SRNet()
    else:
        model = DeblurNet()
    if args.use_gpus:
        model = torch.nn.DataParallel(model.cuda(), device_ids=[device_ids[0]], output_device=device_ids[0])

    model.load_state_dict(torch.load(args.ckp, map_location='cpu'))
    model.eval()

    test_dataset = Dataset_Test(args.TestImgPath, args.TestEvePath)
    test_loader = DataLoader(test_dataset, batch_size=1)

    with torch.no_grad():
        with tqdm(total=len(test_loader)) as pbar:
            for _, items in enumerate(test_loader):
                imgs, event, name = items

                cr = np.uint8(torch.squeeze(imgs[:, 1, :, :]).numpy())
                cb = np.uint8(torch.squeeze(imgs[:, 2, :, :]).numpy())

                imgs = imgs[:, 0:1, :, :].to(device)
                event = event.to(device)
                outs = model(imgs, event)

                im = np.uint8(torch.squeeze(outs.cpu()).numpy())
                im = cv2.cvtColor(cv2.merge([im, cr, cb]), cv2.COLOR_YCrCb2BGR)
                cv2.imwrite(os.path.join(args.TestSavePath, name[0]), im)
                pbar.update()

if __name__ == "__main__":
    test()