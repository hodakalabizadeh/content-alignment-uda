import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from .warping import SpatialTransformer, VecIntegration


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ndims):
        super(ConvBlock, self).__init__()

        self.conv = getattr(nn, f"Conv{ndims}d")(in_channels, out_channels, kernel_size=3, stride = 1, padding = 1)
        self.batchnorm = getattr(nn, f"BatchNorm{ndims}d")(out_channels)
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.batchnorm(out)
        out = self.activation(out)
        return out

class RegistrationUNet(nn.Module):
    def __init__(
        self,
        in_shape = None,
        infeatures = None,
        nb_features = None,
        nb_levels = None,
        max_pool=2,
        nb_conv_per_level=1,

        ):
        super().__init__()
        ndims = len(in_shape)  # 2 = images, 3 = volumes
        assert ndims in (2, 3), "in_shape must be 2D or 3D"

        # extract any surplus (full resolution) decoder convolutions    
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:] # Convolutions for the last k blocks that come after last skip connection
        dec_nf = dec_nf[:nb_dec_convs] # Decoder convolutions are equal in length as encode 
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) # add one for the unconnected in the middle? 

        max_pool = [max_pool] * self.nb_levels # Need a max pool operation per level

        # Define upsampling and pooling operations
        self.pool = [getattr(nn, f"MaxPool{ndims}d")(i) for i in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]


        # ================= Contraction/Encoder path ==================
     
        self.encoder = nn.ModuleList()
        prev_nf = infeatures 
        encoder_nfs = [prev_nf]

        # Loop through every level
        for level in range(self.nb_levels):
            
            # loop through number of convolutions per layer
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                convs.append(ConvBlock(prev_nf, enc_nf[level], ndims))
                prev_nf = enc_nf[level]    

            # You append to encoder from the output of all convolution steps in that layer 
            self.encoder.append(convs)

            encoder_nfs.append(prev_nf)
        
        
        # ================== Expansion/Decoder path =====================
        self.decoder = nn.ModuleList()

        encoder_nfs = np.flip(encoder_nfs)
        
        # Loop through every level
        for level in range(self.nb_levels):
            
            # loop through number of convolutions per layer
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                convs.append(ConvBlock(prev_nf, dec_nf[level], ndims))
                prev_nf = dec_nf[level]   

            # You append to decoder from the output of all convolution steps in that layer 
            self.decoder.append(convs)

            prev_nf += encoder_nfs[level]


        # ====================== Final layers ========================
        self.final_layers = nn.ModuleList()
        
        # Loop through convolutions for the last layers         
        for _, nf in enumerate(final_convs):
            self.final_layers.append(ConvBlock(prev_nf, nf, ndims))
            prev_nf = nf 

        # final number of features
        self.final_nf = prev_nf


     # ====================== Define Model ========================
    def forward(self, x):
        skip_connections = []

        # Encoder path / forward pass
        for level, convs in enumerate(self.encoder):
            
            for conv in convs:
                x = conv(x)
                
            skip_connections.append(x)
            x = self.pool[level](x)
        
        # Reverse the skip connections as they are concatinated from the lowest resolution first
        skip_connections = skip_connections[::-1] 

        # Decoder path
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            x = self.upsampling[level](x)
            x = torch.cat([x, skip_connections[level]], dim=1)

        # final convs at full resoltion
        for conv in self.final_layers:
            x = conv(x)

        return x


class BidirDiffeoRegistration(nn.Module):
    """Bidirectional diffeomorphic registration network (the registration network R in the paper).

    A U-Net takes the source and target images, stacked along the channel axis, and predicts a
    stationary velocity field. Integrating the field (scaling and squaring) gives a diffeomorphic
    source -> target deformation; integrating its negative gives the inverse, target -> source.
    The source image and label are warped to the target, and the target image back to the source.

    Works for 2D images and 3D volumes; the dimensionality is taken from `inshape`.
    Adapted from VoxelMorph (credit: Adrian Dalca and Andrew Hoopes).

    Args:
        inshape: spatial size of the inputs, (H, W) or (H, W, D).
        in_channels: channels per image, e.g. 1 for MRI or 3 for MNIST-M (RGB).
        nb_unet_features: U-Net features as [[encoder features], [decoder features]].
        nb_unet_conv_per_level: convolutions per U-Net level.
        int_steps: scaling-and-squaring steps; 0 gives a non-diffeomorphic warp.
    """

    def __init__(self, inshape, in_channels, nb_unet_features, nb_unet_conv_per_level=1, int_steps=7):
        super().__init__()
        ndims = len(inshape)
        assert ndims in (2, 3), f"inshape must be 2D or 3D, got {ndims}D"

        # U-Net on the stacked source and target images
        self.unet_model = RegistrationUNet(
            in_shape=inshape,
            infeatures=2 * in_channels,
            nb_features=nb_unet_features,
            max_pool=2,
            nb_conv_per_level=nb_unet_conv_per_level,
        )

        # U-Net features -> velocity field (one channel per spatial axis), initialised near zero
        self.flow = getattr(nn, f"Conv{ndims}d")(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # velocity field -> diffeomorphic deformation
        self.integ = VecIntegration(inshape=inshape, nb_steps=int_steps)

        # warping: images and training labels bilinearly, evaluation labels with nearest neighbour
        self.spatial_transformer_img = SpatialTransformer(size=inshape, mode='bilinear')
        self.spatial_transformer_label = SpatialTransformer(size=inshape, mode='bilinear')
        self.spatial_transformer_label_val = SpatialTransformer(size=inshape, mode='nearest')

    def forward(self, source_img, target_img, source_label, train=None):
        """Register the source to the target, and the target back to the source.

        Args:
            source_img, target_img: (B, in_channels, *inshape).
            source_label: one-hot source label, (B, n_classes, *inshape).
            train: how the source label is warped. True uses bilinear interpolation, giving a soft
                label that the segmentation loss can backpropagate through into the registration;
                False uses nearest neighbour, giving a hard label for evaluation.
                Defaults to `self.training` (inherited from `nn.Module`; set by `model.train()` /
                `model.eval()`).

        Returns:
            source_to_target_img, pos_diffeo_field, source_to_target_label,
            target_to_source_img, neg_diffeo_field
        """
        x = self.unet_model(torch.cat([source_img, target_img], dim=1))
        pos_flow = self.flow(x)                     # stationary velocity field
        pos_diffeo_field = self.integ(pos_flow)     # source -> target deformation
        neg_diffeo_field = self.integ(-pos_flow)    # target -> source (inverse) deformation

        source_to_target_img = self.spatial_transformer_img(source_img, pos_diffeo_field)
        target_to_source_img = self.spatial_transformer_img(target_img, neg_diffeo_field)

        if train is None:
            train = self.training
        warp_label = self.spatial_transformer_label if train else self.spatial_transformer_label_val
        source_to_target_label = warp_label(source_label, pos_diffeo_field)

        return source_to_target_img, pos_diffeo_field, source_to_target_label, target_to_source_img, neg_diffeo_field
