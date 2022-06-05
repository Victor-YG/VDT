import torch.nn as nn


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