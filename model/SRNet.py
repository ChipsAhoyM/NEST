from .BaseBlock import *
from .NestEst import EventEncoder
import torch.nn as nn
from .RDDB import *

class SRNet(nn.Module):
    def __init__(self, nf, nb, upscale=4):
        super(SRNet, self).__init__()
        self.nb = nb
        self.fea_conv = nn.Sequential(
            Conv2D(1, 64, 3),
            Dense_block(64, 16)
        )
        self.EventEncoder = nn.Sequential(
            EventEncoder()
        )

        self.rb_blocks = nn.ModuleList()
        self.ev_blocks = nn.ModuleList()
        self.fu_blocks = nn.ModuleList()
        self.att_blocks = SEBlock(128,8)
        for _ in range(nb):
            self.rb_blocks.append(RRDB(nf*2, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', norm_type='batch', act_type='leakyrelu', mode='CNA')) 
            self.fu_blocks.append(conv_block(2*nf, nf, kernel_size=3, norm_type='batch', act_type = 'leakyrelu'))
            self.ev_blocks.append(RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', norm_type='batch', act_type='leakyrelu', mode='CNA'))

        self.upsampler = nn.Sequential(
            pixelshuffle_block(nf, nf, act_type='leakyrelu',norm_type='batch'),
            ResidualBlock(nf),
            pixelshuffle_block(nf, nf, act_type='leakyrelu',norm_type='batch'),
            ResidualBlock(nf),
            conv_block(nf, 1, kernel_size=3, norm_type='batch', act_type=None)
        )
        
        self.up = nn.Upsample(scale_factor=upscale,mode="bilinear",align_corners=False)
        

    def forward(self, x, e):
        x = (x)/255.0
        SRcode = self.EventEncoder(e)
        Srimg = self.makeframe(x,SRcode)
        return Srimg
    
    def makeframe(self,x,event):
        upx = self.up(x)
        x_ori = self.fea_conv(x)
        features = x_ori
        evfeatures = self.att_blocks(event)
        for i in range(self.nb):
            evfeatures = self.ev_blocks[i](evfeatures)
            fuse = self.rb_blocks[i](torch.cat([evfeatures,features],dim=1))
            features = self.fu_blocks[i](fuse)
        fusion = features + x_ori
        upRes = self.upsampler(fusion)

        return torch.clamp(upRes+upx,min=0.0,max=1.0)*255