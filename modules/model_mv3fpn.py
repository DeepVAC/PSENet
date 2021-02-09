'''MobileNetV3 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from deepvac.syszux_mobilenet import MobileNetV3, MobileNetV3Large

class Backbone(MobileNetV3Large):
    def __init__(self):
        super(Backbone, self).__init__(width_mult=1.)

    def initFc(self):
        #self.downsampler_list = [1,3,8]
        # for large
        self.downsampler_list = [3,6,12]
        #self.layers_number = len(self.features)

    def forward(self,x):
        out = []
        for i, fea in enumerate(self.features):
            x = fea(x)
            if i in self.downsampler_list:
                out.append(x)
        x = self.conv(x)
        out.append(x)
        return out

class mobilenetv3(nn.Module):
    def __init__(self, kernel_num=7):
        super(mobilenetv3, self).__init__()
        inplanes = 16
        self.backbone = Backbone()
        #out = [16, 24, 48, 576]
        # for large
        out = [24, 40, 112, 960]
        
        conv_out = 128
        self.toplayer = nn.Sequential(
            nn.Conv2d(out[3], conv_out, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(conv_out),
            nn.ReLU(inplace=True)
        )

        # Lateral layers
        self.latlayer1 = nn.Sequential(
            nn.Conv2d(out[2], conv_out, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(conv_out),
            nn.ReLU(inplace=True)
        )

        self.latlayer2 = nn.Sequential(
            nn.Conv2d(out[1], conv_out, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(conv_out),
            nn.ReLU(inplace=True)
        )

        self.latlayer3 = nn.Sequential(
            nn.Conv2d(out[0], conv_out, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(conv_out),
            nn.ReLU(inplace=True)
        )

        # Smooth layers
        self.smooth1 = nn.Sequential(
            nn.Conv2d(conv_out, conv_out, kernel_size=3, stride=1, padding=1, groups=conv_out),
            nn.BatchNorm2d(conv_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_out, conv_out, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(conv_out),
            nn.ReLU(inplace=True)
        )

        self.smooth2 = nn.Sequential(
            nn.Conv2d(conv_out, conv_out, kernel_size=3, stride=1, padding=1, groups=conv_out),
            nn.BatchNorm2d(conv_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_out, conv_out, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(conv_out),
            nn.ReLU(inplace=True)
        )

        self.smooth3 = nn.Sequential(
            nn.Conv2d(conv_out, conv_out, kernel_size=3, stride=1, padding=1, groups=conv_out),
            nn.BatchNorm2d(conv_out),
            nn.Conv2d(conv_out, conv_out, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(conv_out),
            nn.ReLU(inplace=True)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(conv_out*4, conv_out, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(conv_out),
            nn.ReLU(inplace=True)
        )

        self.out_conv = nn.Conv2d(conv_out, kernel_num, kernel_size=1, stride=1)

        self.init_params()

    def _upsample(self, x, y, scale=1):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:], mode='bilinear') + y

    def _upsample_cat(self, p2, p3, p4, p5):
        h, w = p2.size()[2:]
        p3 = F.interpolate(p3, size=(h, w), mode='bilinear')
        p4 = F.interpolate(p4, size=(h, w), mode='bilinear')
        p5 = F.interpolate(p5, size=(h, w), mode='bilinear')

        return torch.cat((p2, p3, p4, p5), 1)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

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

        out = self._upsample(out, x, scale=1)
        return out

def test():
    net = mobilenetv3()
    #print(net)
    x = torch.randn(2,3,640,640)
    y = net(x)
    print(y.size())

#test()
