import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels * 2, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2),
            nn.ReLU()
        )