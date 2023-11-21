import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class UNet(nn.Module):

    def __init__(self, input_size, kernel_size=(3, 3, 3, 3, 3, 3), output_channels=(8, 16, 32, 64, 128, 256), strides=(2, 2, 2, 2, 2, 2), blur_input=False):
        """
        Unet (add ref)
        Args:
            input_size:  shape of the input image. Should be a 2 element vector for a 2D image.
        """

        super().__init__()

        self.name = 'UNet'
        # This should be in spherical coordinates
        self.input_size = input_size
        self.n_output_channels = (1,) + output_channels  # add the channels of the input image
        self.kernel_size = kernel_size
        self.strides = strides
        self.blur_input =blur_input

        if self.blur_input is True:
            self.blur_filter = self.compute_Gaussian_filter()
        else:
            self.blur_filter = None

        # compute paddings
        self.layer_sizes = (list(input_size),)
        for idx in range(len(self.n_output_channels) - 1):
            newsize = [
                np.floor((s + 2 * np.floor((self.kernel_size[idx] - 1) / 2) - self.kernel_size[idx]) / self.strides[
                    idx] + 1).astype(np.int)
                for s in self.layer_sizes[idx]]
            self.layer_sizes = self.layer_sizes + (newsize,)

        # layer_size = (np.array(input_size[1:]),)  # todo hardcoded for dim = 0. Other dimensions need to be re-implemented
        # for i in range(len(self.kernel)-1):
        #     layer_size += (np.array(np.ceil(layer_size[i]), dtype=np.int),)

        # define the "encoder". First a few convs and then layers are: strided conv - conv cond

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels=self.n_output_channels[0], out_channels=self.n_output_channels[1],
                      kernel_size=self.kernel_size[0], stride=1, padding=1),
            nn.BatchNorm2d(self.n_output_channels[1]),
            nn.ReLU(),

            #
            nn.Conv2d(in_channels=self.n_output_channels[1], out_channels=self.n_output_channels[1],
                      kernel_size=self.kernel_size[1], stride=1, padding=1),
            nn.BatchNorm2d(self.n_output_channels[1]),
            nn.ReLU(),

        )
        # encoder downsample -> conv
        layers = []
        for i in range(1, len(self.kernel_size)):
            current_layer = nn.Sequential(

                nn.Conv2d(in_channels=self.n_output_channels[i], out_channels=self.n_output_channels[i],
                          kernel_size=self.kernel_size[i], stride=self.strides[i], padding=1),
                nn.BatchNorm2d(self.n_output_channels[i]),
                nn.ReLU(),

                #
                nn.Conv2d(in_channels=self.n_output_channels[i], out_channels=self.n_output_channels[i + 1],
                          kernel_size=self.kernel_size[i], stride=1, padding=1),
                nn.BatchNorm2d(self.n_output_channels[i + 1]),
                nn.ReLU(),

            )
            layers.append(current_layer)
        self.encoder = nn.Sequential(*layers)

        # bridge: upsample
        p = [s - ((self.layer_sizes[-1][idx + 1] - 1) * self.strides[-1] - 2 * np.floor(
            (self.kernel_size[-1] - 1) / 2).astype(np.int) + self.kernel_size[-1])
             for idx, s in enumerate(self.layer_sizes[len(self.kernel_size) - 1][1:])]
        self.bridge = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.n_output_channels[-1],
                               out_channels=self.n_output_channels[-2],
                               kernel_size=self.kernel_size[-1], stride=self.strides[-1], padding=1,
                               output_padding=p),
            nn.BatchNorm2d(self.n_output_channels[-2]),
            nn.ReLU(),
        )

        # decoder: goes conv *2, conv,  upsample,
        layers = []
        for i in reversed(range(len(self.kernel_size) - 1)):
            if i == 0:
                upsampler = nn.Identity()
            else:
                sz = self.layer_sizes[i - 1] if len(self.layer_sizes[i - 1]) == 2 else self.layer_sizes[i - 1][1:]
                upsampler = nn.Upsample(size=sz, mode='nearest')

            current_layer = nn.Sequential(
                nn.Conv2d(in_channels=self.n_output_channels[i + 1] * 2, out_channels=self.n_output_channels[i + 1],
                          kernel_size=self.kernel_size[i], stride=1, padding=1),
                nn.BatchNorm2d(self.n_output_channels[i + 1]),
                nn.ReLU(),
                # standard conv
                nn.Conv2d(in_channels=self.n_output_channels[i + 1], out_channels=self.n_output_channels[i],
                          kernel_size=self.kernel_size[i], stride=1, padding=1),
                nn.BatchNorm2d(self.n_output_channels[i]),
                nn.ReLU(),
                # upsample and decrease n channels
                upsampler,
                nn.BatchNorm2d(self.n_output_channels[i]),
                nn.ReLU(),

            )
            layers.append(current_layer)
        self.decoder = nn.Sequential(*layers)
        # output layer
        self.output_layer = nn.Sequential(
            nn.Conv2d(in_channels=self.n_output_channels[0], out_channels=self.n_output_channels[0],
                      kernel_size=self.kernel_size[0], stride=1, padding=1),
            nn.Sigmoid(),
        )

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.blur_filter = self.blur_filter.to(*args, **kwargs)
        return self

    def compute_Gaussian_filter(self, sigma=1.5):
        # Calculate the size of the Gaussian kernel
        k_size = int(2 * 4 * sigma + 1)  # Adjust the multiplier (4) as needed

        # Create a 1-dimensional Gaussian kernel
        kernel_1d = np.exp(-0.5 * (np.arange(-k_size, k_size + 1) / sigma) ** 2)
        kernel_1d = kernel_1d / np.sum(kernel_1d)  # Normalize the kernel

        # Convert the 1-dimensional kernel to a 2-dimensional kernel
        kernel_2d = np.outer(kernel_1d, kernel_1d)

        # Convert the NumPy kernel to a PyTorch tensor
        kernel_tensor = torch.tensor(kernel_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return kernel_tensor

    def extra_repr(self):
        out_string = 'input_size={}'.format(self.input_size)
        return out_string

    def forward(self, data):

        if self.blur_input is True:
            # Define your hardcoded filter/kernel
            # Apply the convolution using the conv2d function
            data = F.conv2d(data, self.blur_filter, padding='same')

        feat = self.input_layer(data)
        features = [feat]
        for i, layer in enumerate(self.encoder):
            feat = layer(feat)
            if i < len(self.encoder) - 1:
                features.append(feat)

        # bridge
        feat = self.bridge(feat)

        # decoder (summed skip connections)
        for id in reversed(range(len(self.decoder))):
            skip_i = features[id]
            # concat = feat + skip_i
            concat = torch.cat((feat, skip_i), dim=1)
            feat = self.decoder[len(self.decoder) - id - 1](concat)

        y = self.output_layer(feat)
        return y

    def get_name(self):
        #linear_feat_str = '_features{}'.format(self.linear_features).replace(', ', '_').replace('(', '').replace(')', '')
        return self.name + '{}'.format('_blurred' if self.blur_input is True else '') #+ linear_feat_str
