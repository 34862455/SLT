from signjoey.VGGNET import *
import torch
from torchvision import transforms
from signjoey.P3DNet import *


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # call
        # defined in vggnet.py?
        #  Extracts spatial features
        self.vgg = Vgg(n_classes=64)
        # call
        # defined in p3dnet.py?
        # Extracts spatiotemporal features
        self.p3d = P3D63(layers=[3, 4, 6, 3],input_channel=3,  #(self.nChannels//2)+3*self.growthRate+1
                                    num_classes=1024)


    # Reorders dimensions to match the expected input format
    def forward(self, x):
        x = x.permute(0,2,3,4,1).contiguous()
        batch, _, w, h, channel = x.shape

        out = self.vgg(x)
        # Splits the sequence of frames into overlapping 16-frame clips (window size = 16, step size = 8)
        out = out.unfold(dimension=1, size=16, step=8).contiguous()
        bs, ts , _, _ = out.shape
        out = out.view(bs,ts,-1)

        # Changes shape to (batch, frames, channels, height, width), making it compatible with P3D
        x = x.permute(0, 1, 4, 2, 3)
        # Splits into overlapping 16-frame clips
        x = x.unfold(dimension=1, size=16, step=8).permute(0,1,2,5,3,4).contiguous()

        # Flattens the batch & time dimension
        batch, time_step, channel, clip_length, w, h = x.shape
        x = x.view(batch * time_step, channel, clip_length, h, w)
        # Passes the clips through P3D
        # Extracts 1024-dimensional spatiotemporal features
        outp = self.p3d(x)
        outp = outp.view(batch,time_step,-1)
        # Fuses features from both networks by summing
        return torch.add(outp,out)