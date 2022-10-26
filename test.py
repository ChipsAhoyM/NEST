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
else:
    device = torch.device("cpu")

def test():
    os.makedirs(args.TestSavePath, exist_ok=True)
    if args.mode == 'sr':
        model = SRNet()
    else:
        model = DeblurNet()
    if args.use_gpus:
        model = model.to(device)


    weights_name = args.ckp
    weights = torch.load(weights_name)
    
    weights_dict = {}
    for k, v in weights.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v

    model.load_state_dict(weights_dict)
    model.eval()

    test_dataset = Dataset_Test(args.TestImgPath, args.TestEvePath, args.upsample_scale)
    test_loader = DataLoader(test_dataset, batch_size=1)

    with torch.no_grad():
        with tqdm(total=len(test_loader)) as pbar:
            for data in test_loader:
                imgs = data['input'][:, 0:1, :, :].to(device)
                event = data['ev'].to(device)
                outs = model(imgs, event)

                umsampled = data['upsample']
                cr = np.uint8(torch.squeeze(umsampled[:, 1, :, :]).numpy())
                cb = np.uint8(torch.squeeze(umsampled[:, 2, :, :]).numpy())
                im = np.uint8(torch.squeeze(outs.cpu()).numpy())
                im = cv2.cvtColor(cv2.merge([im, cr, cb]), cv2.COLOR_YCrCb2BGR)
                cv2.imwrite(os.path.join(args.TestSavePath, data['name'][0] + f"_{args.mode}.png"), im)
                pbar.update()
    
    torch.save(model.state_dict(), args.ckp)

if __name__ == "__main__":
    test()