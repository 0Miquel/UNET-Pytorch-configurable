import torch.nn as nn
import torch


class UNET(nn.Module):
    def __init__(self, in_channels, out_channels, features=32, n_blocks=3, kernel_size=3):
        super().__init__()

        encoder = []
        for i in range(n_blocks):
            encoder.append(self.encode_block(in_channels, features, kernel_size=kernel_size, padding=kernel_size//2))
            in_channels = features
            features = features * 2
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        features = in_channels // 2
        for i in range(n_blocks - 1):
            decoder.append(self.decode_block(in_channels, features, kernel_size=kernel_size, padding=kernel_size//2))
            in_channels = features * 2
            features = features // 2
        decoder.append(self.decode_block(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        # downsampling part
        residuals = []
        for conv in self.encoder:
            x = conv(x)
            # save
            residuals.append(x)
        # tract residuals
        residuals = residuals[:-1]
        residuals.reverse()
        # upsampling part
        upconv = self.decoder[0]
        x = upconv(x)
        for upconv, residual in zip(self.decoder[1:], residuals):
            x = upconv(torch.cat([x, residual], 1))

        return x

    def encode_block(self, in_channels, out_channels, kernel_size, padding):
        encode = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        return encode

    def decode_block(self, in_channels, out_channels, kernel_size, padding):
        decode = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        return decode



