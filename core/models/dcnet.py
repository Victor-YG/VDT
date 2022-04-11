'''
This is my implementation of depth completion network.
Its structure resembles Resnet50.
'''

from turtle import forward
import torch
import torch.nn as nn


class DCnet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, latent_space=512):
        super(DCnet, self).__init__()

        SIZE_0 = in_channels
        SIZE_1 = latent_space // 16
        SIZE_2 = latent_space //  8
        SIZE_3 = latent_space //  4
        SIZE_4 = latent_space //  2
        SIZE_5 = latent_space

        self.conv_1 = nn.Conv2d(SIZE_0, SIZE_1, kernel_size=9, padding="same")

        self.encoding_block_1 = EncodingBlock(SIZE_1, SIZE_2)
        self.encoding_block_2 = EncodingBlock(SIZE_2, SIZE_3)
        self.encoding_block_3 = EncodingBlock(SIZE_3, SIZE_4)
        self.encoding_block_4 = EncodingBlock(SIZE_4, SIZE_5)

        self.decoding_block_1 = DecodingBlock(SIZE_5, SIZE_4)
        self.decoding_block_2 = DecodingBlock(SIZE_4, SIZE_3)
        self.decoding_block_3 = DecodingBlock(SIZE_3, SIZE_2)
        self.decoding_block_4 = DecodingBlock(SIZE_2, SIZE_1)

        self.conv_2 = nn.Conv2d(SIZE_1, out_channels, kernel_size=3, padding="same")

    def name(self): return "dcnet"

    def forward(self, c, d):
        x = torch.cat((c, d), dim=1)
        y_0 = self.conv_1(x)

        # encoding
        y_1 = self.encoding_block_1(y_0)
        y_2 = self.encoding_block_2(y_1)
        y_3 = self.encoding_block_3(y_2)
        y   = self.encoding_block_4(y_3)
        
        # decoding
        y = self.decoding_block_1(y) + y_3
        y = self.decoding_block_2(y) + y_2
        y = self.decoding_block_3(y) + y_1
        y = self.decoding_block_4(y) + y_0
        y = self.conv_2(y)
        y = y + d

        return y


class EncodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncodingBlock, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding="same")
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding="same")
        self.conv_3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding="same")

        self.max_pool = nn.MaxPool2d((2, 2))
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv_1(x)
        y = self.relu(y)
        y = self.conv_2(y)
        y = self.relu(y)
        y = self.conv_3(y)
        y = self.max_pool(y)
        y = self.relu(y)
        return y



class DecodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecodingBlock, self).__init__()

        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.conv_1 = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=1, padding="same")
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=1, padding="same")
        self.conv_3 = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=1, padding="same")
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.deconv(x)
        y = self.conv_1(y)
        y = self.relu(y)
        y = self.conv_2(y)
        y = self.relu(y)
        y = self.conv_3(y)
        y = self.relu(y)
        return y


if __name__ == "__main__":
    model = DCnet()
