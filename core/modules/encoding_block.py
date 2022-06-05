import torch.nn as nn


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