'''MobileNetV3 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepvac.backbones import MobileNetV3, MobileNetV3Large, Conv2dBNReLU, Concat

class Backbone(MobileNetV3Large):
    def __init__(self):
        super(Backbone, self).__init__(width_mult=1.)

    def initFc(self):
        self.downsampler_list = [3,6,12]

    def forward(self,x):
        out = []
        for i, fea in enumerate(self.features):
            x = fea(x)
            if i in self.downsampler_list:
                out.append(x)
        x = self.conv(x)
        out.append(x)
        return out

class FpnMobileNetv3(nn.Module):
    def __init__(self, kernel_num=7):
        super(FpnMobileNetv3, self).__init__()
        inplanes = 16
        self.backbone = Backbone()
        out = [24, 40, 112, 960]
        
        conv_out = 128
        self.toplayer = Conv2dBNReLU(out[3],conv_out,kernel_size=1,stride=1,padding=0)

        # Lateral layers
        self.latlayer1 = Conv2dBNReLU(out[2], conv_out, kernel_size=1, stride=1, padding=0)

        self.latlayer2 = Conv2dBNReLU(out[1], conv_out, kernel_size=1, stride=1, padding=0)

        self.latlayer3 = Conv2dBNReLU(out[0], conv_out, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1 = nn.Sequential(
            Conv2dBNReLU(conv_out, conv_out, kernel_size=3, stride=1, padding=1, groups=conv_out),
            Conv2dBNReLU(conv_out, conv_out, kernel_size=1, padding=0, stride=1)
        )

        self.smooth2 = nn.Sequential(
            Conv2dBNReLU(conv_out, conv_out, kernel_size=3, stride=1, padding=1, groups=conv_out),
            Conv2dBNReLU(conv_out, conv_out, kernel_size=1, padding=0, stride=1)
        )

        self.smooth3 = nn.Sequential(
            Conv2dBNReLU(conv_out, conv_out, kernel_size=3, stride=1, padding=1, groups=conv_out),
            Conv2dBNReLU(conv_out, conv_out, kernel_size=1, padding=0, stride=1)
        )

        self.conv = Conv2dBNReLU(conv_out*4, conv_out, kernel_size=3, padding=1, stride=1)

        self.out_conv = nn.Conv2d(conv_out, kernel_num, kernel_size=1, stride=1)
        self.cat = Concat()

    def _upsample(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear')

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:], mode='bilinear') + y

    def _upsample_cat(self, p2, p3, p4, p5):
        h, w = p2.size()[2:]
        p3 = F.interpolate(p3, size=(h, w), mode='bilinear')
        p4 = F.interpolate(p4, size=(h, w), mode='bilinear')
        p5 = F.interpolate(p5, size=(h, w), mode='bilinear')
        return self.cat([p2, p3, p4, p5])

    def forward(self, x):
        c2, c3, c4, c5 = self.backbone(x)
        
        # Head
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p4 = self.smooth1(p4)
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p3 = self.smooth2(p3)
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        p2 = self.smooth3(p2)

        out = self._upsample_cat(p2, p3, p4, p5)
        out = self.conv(out)
        out = self.out_conv(out)

        out = self._upsample(out, x)
        return out

def test():
    net = FpnMobileNetv3()
    x = torch.randn(2,3,630,640)
    y = net(x)
    print(y.size())

#test()
