'''
This is my implementation of depth completion network.
 - Input is stacked left (rgb) and depth image (in_channels = 4)
 - Output is the completed and refined depth image and uncertainty score (out_channels = 2)
'''

import torch
import torch.nn as nn

from .modules.encoding_block import EncodingBlock
from .modules.decoding_block import DecodingBlock


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
        '''
        run depth completion network
        c - input color iamge
        d - input depth image
        '''
        
        x = torch.cat((c, d), dim=1)
        y_0 = self.conv_1(x)

        # encoding
        y_1 = self.encoding_block_1(y_0)
        y_2 = self.encoding_block_2(y_1)
        y_3 = self.encoding_block_3(y_2)
        y   = self.encoding_block_4(y_3)

        # decoding
        y = self.decoding_block_1(y)
        y = nn.functional.interpolate(input=y, size=[y_3.shape[2], y_3.shape[3]], mode="bilinear") + y_3
        y = self.decoding_block_2(y)
        y = nn.functional.interpolate(input=y, size=[y_2.shape[2], y_2.shape[3]], mode="bilinear") + y_2
        y = self.decoding_block_3(y)
        y = nn.functional.interpolate(input=y, size=[y_1.shape[2], y_1.shape[3]], mode="bilinear") + y_1
        y = self.decoding_block_4(y)
        y = nn.functional.interpolate(input=y, size=[y_0.shape[2], y_0.shape[3]], mode="bilinear") + y_0
        y = self.conv_2(y)
        y = y + d

        return y


if __name__ == "__main__":
    model = DCnet()
