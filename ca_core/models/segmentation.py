import torch
import torch.nn as nn

class DoubleConvBlock(nn.Module):
    def __init__(self, ndims, in_channels, out_channels):
        super().__init__()
        Conv = getattr(nn, 'Conv%dd' % ndims)
        batchNorm = getattr(nn, 'BatchNorm%dd' % ndims)
        self.conv = nn.Sequential(
            # same convolution: input dims = output dims
            Conv(in_channels, out_channels, kernel_size = 3, padding = 'same', bias = False),
            batchNorm(num_features=out_channels),
            nn.ReLU(inplace = True),

            Conv(out_channels, out_channels, kernel_size = 3, padding = 'same', bias = False),
            batchNorm(num_features=out_channels),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        return self.conv(x)
    

class SegmentationUNet(nn.Module):
    def __init__(
        self,
        in_shape = None,
        infeatures = None,
        outfeatures = None,
        nb_features = None,
        max_pool=2,
        batchNorm_reg = False,
        dropout_reg = False
        ):
        super().__init__()

        # ensure correct dimensionality
        ndims = len(in_shape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pool = MaxPooling(kernel_size = max_pool)

        Transpose = getattr(nn, 'ConvTranspose%dd' % ndims)
        

        # Contraction/Encoder path 
        for feature in nb_features:
            self.downs.append(DoubleConvBlock(ndims, infeatures, feature))
            infeatures = feature

        # Expansion/Decoder path 
        for feature in reversed(nb_features):
            self.ups.append(
                # Up-convolution 
                Transpose(
                    in_channels = feature*2, out_channels=feature, kernel_size = 2, stride = 2
                )
            )
            self.ups.append(DoubleConvBlock(ndims, feature*2, feature))
        
        self.bottleneck = DoubleConvBlock(ndims, nb_features[-1], nb_features[-1]*2)
        
        # Add a final convolutional layer for output (here with a one chanel output)
        conv = getattr(nn, 'Conv%dd' % ndims)
        self.final_conv = conv(nb_features[0], outfeatures, kernel_size=1)
        self.activation = nn.Softmax(dim=1)
 
    def forward(self,x):
        skip_connections = []

        # Go through encoder path and add results to the skip connections
        for down in self.downs: # Go through each section of encoder path one by one
            x = down(x) 
            skip_connections.append(x)
            x = self.pool(x) # max pool 2x2
        
        # Go through bottle neck 
        x = self.bottleneck(x)

        # Reverse the skip connections as they are concatinated from the lowest resolution first
        skip_connections = skip_connections[::-1]

        # Treating the decoder path as two steps (upconv + 3x3 conv) therefore obtaining the result of 
        # the upconv only
        for idx in range(0, len(self.ups), 2):
            # Fist pass through up-conv
            x = self.ups[idx](x)

            # Obtain result from encoder path using skip connection
            skip_connection = skip_connections[idx//2] 
            
            # Resize if the dims are not the same (original U-net paper crops this)
            x = nn.functional.interpolate(x, size = skip_connection.shape[2:])

            # Concat the skip connection with the up-conv result
            concat_skip = torch.cat((skip_connection,x), dim = 1)

            # Pass through the double conv the concatanated results
            x = self.ups[idx+1](concat_skip)
        
        # Return output of final layer
        final = self.final_conv(x)
        y = self.activation(final)
        return y
