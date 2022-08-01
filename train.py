import os
import time
import torch
import numpy as np
import skimage.metrics
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

from model import DeblurNet,SRNet,myVGG19
from utils import Options, Dataset_Train, Logger, init_model

args = Options().parse()
if args.use_gpus:
    device = torch.device("cuda")
    device_ids = [Id for Id in range(torch.cuda.device_count())]
else:
    device = torch.device("cpu")
vgg19 = myVGG19().to(device)
vgg19.eval()


def loss(outputs, gt):
    alpha = 0.5
    beta = 200.0
    outputs = torch.cat([outputs, outputs, outputs], dim=1)
    gt = torch.cat([gt, gt, gt], dim=1)
    perceptual_X = vgg19(outputs)
    perceptual_Y = vgg19(gt)
    p_loss = torch.nn.MSELoss()(outputs, gt)
    p_f_loss = 0
    for f_x, f_y in zip(perceptual_X, perceptual_Y):
        p_f_loss += torch.mean(torch.sub(f_x, f_y) ** 2)

    content_loss = alpha * p_f_loss + beta * p_loss
    return content_loss


def make_dataset():
    all_dataset = Dataset_Train(args.TrainImgPath, args.TrainEvePath, args.TrainGTPath, args.CropSize, args.mode)
    val_size = int(len(all_dataset) * args.split_scale)
    train_dataset, val_dataset = random_split(all_dataset, [len(all_dataset) - val_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True)
    return train_loader, val_loader


def train():
    job_time = time.strftime("%d_%m_%Y_%H_%M_%S")
    path_log = os.path.join(args.LogPath, job_time)
    logger = Logger(path_log, force=True)
    os.makedirs(path_log, exist_ok=True)
    if args.mode == 'deblur':  
        model = DeblurNet()
    elif args.mode == 'sr':
        model = SRNet()
    else:
        raise NotImplementedError('Model Error!')
        
    if args.use_gpus:
        model = torch.nn.DataParallel(model.cuda(), device_ids=device_ids, output_device=device_ids[0])
    if args.load_weight:
        model.load_state_dict(torch.load(args.ckp, map_location='cpu'))
    else:
        model = init_model(model)
    criterion = loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.milestone, gamma=args.gamma)
    model.train()

    train_loader, val_loader = make_dataset()
    bestPNSR = 0.0
    for epoch in range(args.BegEpoch, args.NumEpoch + 1):
        epoch_loss = 0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{args.NumEpoch}', dynamic_ncols=True) as pbar:
            for index, items in enumerate(train_loader):
                imgs, event, gt = items
                imgs = imgs.to(device)
                event = event.to(device)
                gt = gt.to(device)


                outs = model(imgs, event)
                Loss = criterion(outs, gt)
                
                optimizer.zero_grad()
                Loss.backward()
                optimizer.step()

                epoch_loss += Loss.item()
                pbar.set_postfix(**{'loss (batch)': Loss.item(), 'AveLoss': epoch_loss / (index + 1)})
                pbar.update()
        torch.save(model.state_dict(), f'pretrained/model_{args.mode}_{epoch}.pth')
        scheduler.step()

        AvePSNR, AveSSIM = validation(model, val_loader, epoch)
        if AvePSNR > bestPNSR:
            torch.save(model.state_dict(), f'pretrained/model_{args.mode}_best.pth')
            bestPNSR = AvePSNR
        logger.add_scalar('validation psnr', AvePSNR)
        logger.add_scalar('validation ssim', AveSSIM)

        logger.add_scalar('loss', epoch_loss/len(train_loader))
        logger.step()

    return model


def validation(model, val_loader, epoch):
    model.eval()
    totally_psnr = 0.
    totally_ssim = 0.

    with torch.no_grad():
        with tqdm(total=len(val_loader), desc=f'Epoch {epoch} validation', dynamic_ncols=True) as pbar:
            for index, items in enumerate(val_loader):
                imgs, event, gt = items
                imgs = imgs.to(device)
                event = event.to(device)

                outs = model(imgs, event)
                img_z = np.uint8(torch.squeeze(outs.cpu()).numpy())
                gt = np.uint8(torch.squeeze(gt).numpy())

                psnr = sum([skimage.metrics.peak_signal_noise_ratio(img_z[i, :, :], gt[i, :, :])
                            for i in range(img_z.shape[0])]) / img_z.shape[0]
                ssim = sum([skimage.metrics.structural_similarity(img_z[i, :, :], gt[i, :, :])
                            for i in range(img_z.shape[0])]) / img_z.shape[0]

                totally_psnr += psnr
                totally_ssim += ssim
                pbar.set_postfix(**{'psnr': psnr, 'ssim': ssim, 'AvePSNR': totally_psnr / (index + 1),
                                    'AveSSIM': totally_ssim / (index + 1)})
                pbar.update()
    model.train()

    return totally_psnr / len(val_loader), totally_ssim / len(val_loader)

if __name__ == '__main__':
    train()
