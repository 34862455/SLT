from signjoey.VGGNET import *
import torch
from torchvision import transforms
from signjoey.P3DNet import *


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.vgg = Vgg(n_classes=64)
        self.p3d = P3D63(layers=[3, 4, 6, 3],input_channel=3,  #(self.nChannels//2)+3*self.growthRate+1
                                    num_classes=1024)


    def forward(self, x):
        x = x.permute(0,2,3,4,1).contiguous()
        batch, _, w, h, channel = x.shape

        out = self.vgg(x)
        out = out.unfold(dimension=1, size=16, step=8).contiguous()
        bs, ts , _, _ = out.shape
        out = out.view(bs,ts,-1)

        x = x.permute(0, 1, 4, 2, 3)
        x = x.unfold(dimension=1, size=16, step=8).permute(0,1,2,5,3,4).contiguous()

        batch, time_step, channel, clip_length, w, h = x.shape
        x = x.view(batch * time_step, channel, clip_length, h, w)
        outp = self.p3d(x)
        outp = outp.view(batch,time_step,-1)
        return torch.add(outp,out)