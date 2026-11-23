
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d, ndims):
        super(Discriminator, self).__init__()
        assert ndims in (2, 3), "ndims must be 2 (images) or 3 (volumes)"
        self.ndims = ndims
        Conv = getattr(nn, f"Conv{ndims}d")
        self.disc = nn.Sequential(
            Conv(
                channels_img,
                features_d,
                kernel_size=2,
                stride=2,
                padding=1
            ), 
            nn.LeakyReLU(0.2),
            self.__block(features_d, features_d*2, 3, 2, 1), 
            self.__block(features_d*2, features_d*4, 3, 2, 1),
            self.__block(features_d*4, features_d*8, 3, 2, 1), 
            Conv(
                features_d*8,
                1,
                kernel_size=2,
                stride=2,
                padding=0
            ), 
            nn.Sigmoid(),
        )

    def __block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            getattr(nn, f"Conv{self.ndims}d")(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias = False,
            ),
            getattr(nn, f"BatchNorm{self.ndims}d")(out_channels),
            nn.LeakyReLU(0.2),
        )
    

    def forward(self, x):
        return self.disc(x)



def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d,
                          nn.Conv3d, nn.ConvTranspose3d, nn.BatchNorm3d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)



