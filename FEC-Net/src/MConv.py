import torch
import torch.nn as nn
from torch import Tensor


class MConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 kernel_size=3,
                 n_div: int = 16,
                 # n_divv: int = 4,
                 n_kk: int = 2,
                 forward: str = 'split_cat'):
        super(MConv2d, self).__init__()
        assert in_channels > 16, "in_channels should > 16, but got {} instead.".format(in_channels)
        self.in_channels = in_channels

        self.conv = nn.Conv2d(in_channels=self.in_channels//4,
                              out_channels=self.in_channels//4,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=(kernel_size - 1) // 2,
                              bias=False)
        self.conv10 = nn.Conv2d(in_channels=(self.in_channels//4)*2,
                                out_channels=(self.in_channels//4)//2,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False)
        self.conv11 = nn.Conv2d(in_channels=(self.in_channels//4)//2,
                                out_channels=(self.in_channels//4)//2,
                                kernel_size=kernel_size,
                                stride=1,
                                padding=(kernel_size - 1) // 2,
                                groups=(self.in_channels//4)//2,
                                bias=False)
        self.conv12 = nn.Conv2d(in_channels=(self.in_channels//4)//2,
                                out_channels=(self.in_channels//4)*2,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing_mconv

        elif forward == 'split_cat':
            self.forward = self.forward_split_cat_mconv

        else:
            raise NotImplementedError("forward method: {} is not implemented.".format(forward))

    def forward_slicing_mconv(self, x: Tensor) -> Tensor:

        x[:, (self.in_channels//16)*1:(self.in_channels//16)*2, :, :] = self.conv(x[:, (self.in_channels//16)*1:(self.in_channels//16)*2, :, :])
        x[:, (self.in_channels//16)*5:(self.in_channels//16)*6, :, :] = self.conv(x[:, (self.in_channels//16)*5:(self.in_channels//16)*6, :, :])
        x[:, (self.in_channels//16)*9:(self.in_channels//16)*10, :, :] = self.conv(x[:, (self.in_channels//16)*9:(self.in_channels//16)*10, :, :])
        x[:, (self.in_channels//16)*13:(self.in_channels//16)*14, :, :] = self.conv(x[:, (self.in_channels//16)*13:(self.in_channels//16)*14, :, :])

        return x

    def forward_split_cat_mconv(self, x: Tensor) -> Tensor:

        x1, x2, x3, x4 = torch.split(x, [self.in_channels // 4, self.in_channels // 4, self.in_channels // 4, self.in_channels // 4], dim=1)
        a1, a2, a3, a4 = torch.split(x1, [self.in_channels // 16, self.in_channels // 16, self.in_channels // 16, self.in_channels // 16], dim=1)
        b1, b2, b3, b4 = torch.split(x2, [self.in_channels // 16, self.in_channels // 16, self.in_channels // 16, self.in_channels // 16], dim=1)
        c1, c2, c3, c4 = torch.split(x3, [self.in_channels // 16, self.in_channels // 16, self.in_channels // 16, self.in_channels // 16], dim=1)
        d1, d2, d3, d4 = torch.split(x4, [self.in_channels // 16, self.in_channels // 16, self.in_channels // 16, self.in_channels // 16], dim=1)
        x = torch.cat((a2, b2, c2, d2), dim=1)
        # D = torch.cat((a1, a3, a4, b1, b3, b4, c1, c3, c4, d1, d3, d4), dim=1)
        D = torch.cat((a1, a4, b1, b4, c1, c4, d1, d4), dim=1)
        S = torch.cat((a3, b3, c3, d3), dim=1)
        x = self.conv(x)
        D = self.conv10(D)
        D = self.conv11(D)
        D = self.conv12(D)
        y = torch.cat((x, S, D), dim=1)
        return y
