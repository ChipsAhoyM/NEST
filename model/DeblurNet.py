from .BaseBlock import *
from .NestEst import EventEncoder

class DeblurNet(nn.Module):
    def __init__(self):
        super(DeblurNet, self).__init__()
    # conv-event
        self.EventEncoder = EventEncoder()
    # conv-img:
        self.conv_1 = nn.Sequential(
            Conv2D(1, 64, 3),
            Dense_block(64, 16)
        )
        self.conv_2 = nn.Sequential(
            Conv2D(128, 128, 2, 2, padding=0),
            Dense_block(128, 32)
        )
        self.conv_3 = nn.Sequential(
            Conv2D(256, 256, 2, 2, padding=0),
            Dense_block(256, 64)
        )
        self.conv_1e = SEBlock(128,8)
        self.conv_2e = nn.Sequential(
            Conv2D(128, 128, 2, 2, padding=0),
            Dense_block(128, 32)
        )
        self.conv_3e = nn.Sequential(
            Conv2D(256, 256, 2, 2, padding=0),
            Dense_block(256, 64)
        )

        self.fusion = nn.Sequential(
            Conv2D(512*2, 512, 1, padding=0),
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512)
        )
    # deconv 
        self.deconv_2 = DeConv2D(512, 256)

        self.conv_5 = nn.Sequential(
            Conv2D(256*3, 128, 1, padding=0),
            Dense_block(128, 32)
        )

        self.deconv_1 = DeConv2D(256, 128)

        self.conv_6 = nn.Sequential(
            Conv2D(128*3, 32, 1, padding=0),
            Dense_block(32, 8)
        )

    # prediction
        self.predConv = nn.Sequential(
            ResidualBlock(channel_num=64),
            ResidualBlock(channel_num=64),
            nn.Conv2d(64, 1, 5, padding=2),
        )

        self.tanh = nn.Tanh()

    def forward(self, image, event):
        inputs = image*2/255.0 - 1.0
        lstm = self.conv_1e(self.EventEncoder(event))
        
        c1 = self.conv_1(inputs)
        c2 = self.conv_2(c1)
        c3 = self.conv_3(c2)

        ce1 = self.conv_2e(lstm)
        ce2 = self.conv_3e(ce1)

        m3 = torch.cat([c3, ce2], dim=1)
        fusion = self.fusion(m3)
        
        dc2 = self.deconv_2(fusion)  
        m2 = torch.cat([c2, ce1, dc2], dim=1)
        c5 = self.conv_5(m2)
        
        dc1 = self.deconv_1(c5)
        m1 = torch.cat([c1, lstm, dc1], dim=1)
        c6 = self.conv_6(m1)

        pred = self.predConv(c6)
        ld_image = self.tanh(pred)
        out = torch.clamp(inputs+ld_image, min=-1.0, max=1.0)

        return (out + 1.0) * 255.0 / 2.0
