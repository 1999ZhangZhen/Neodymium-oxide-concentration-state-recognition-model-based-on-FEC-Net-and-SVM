import math
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.copy_unet import Conv


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            # nn.GELU()
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)              # 张震
            # self.UP_CARAFE = CARAFE(c = in_channels // 2)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # x1 = self.UP_CARAFE(x1)
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

# ---------------------------------------最原始的unet--------------------------------
class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return {"out": logits}

# ----------------------------------------添加了CBAM注意力机制----------------------------------------
# ------------------方案1----------------------------
class UNet_CBAM(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet_CBAM, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)
        self.cbam1 = CBAM(base_c)
        self.cbam2 = CBAM(base_c * 2)
        self.cbam3 = CBAM(base_c * 4)
        self.cbam4 = CBAM(base_c * 8)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x4 = self.cbam4(x4)    # zz
        x = self.up1(x5, x4)
        x3 = self.cbam3(x3)    # zz
        x = self.up2(x, x3)
        x2 = self.cbam2(x2)    # zz
        x = self.up3(x, x2)
        x1 = self.cbam1(x1)  # zz
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return {"out": logits}

# ------------------------方案2----------------------------

class UNet_CBAM2(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet_CBAM2, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)
        self.cbam1 = CBAM(base_c)
        self.cbam2 = CBAM(base_c * 2)
        self.cbam3 = CBAM(base_c * 4)
        self.cbam4 = CBAM(base_c * 8)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x1 = self.cbam1(x1)
        x2 = self.down1(x1)
        x2 = self.cbam2(x2)
        x3 = self.down2(x2)
        x3 = self.cbam3(x3)
        x4 = self.down3(x3)
        x4 = self.cbam4(x4)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return {"out": logits}



# ---------------------方案7（Skip_enhance）--------------------------------
class UNet_Skip_enhance(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet_Skip_enhance, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)
        # self.CARAFE1 = CARAFE(base_c * 2, scale=2)
        # self.CARAFE2 = CARAFE(base_c * 4, scale=4)
        # self.CARAFE3 = CARAFE(base_c * 8, scale=8)
        # 跳跃加强模块1中的上采样
        self.upz1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upz2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upz3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.enhance_Conv1 = Conv(base_c * 12, base_c * 4, k=3, s=1, p=1, g=1, act=nn.ReLU())
        self.enhance_Conv2 = Conv(base_c * 14, base_c * 2, k=3, s=1, p=1, g=1, act=nn.ReLU())
        self.enhance_Conv3 = Conv(base_c * 15, base_c * 1, k=3, s=1, p=1, g=1, act=nn.ReLU())

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x_out1 = self.up1(x5, x4)
# -----------------第三个enhance module----------------------
        x4_up2 = self.upz1(x4)
        x3_add = torch.cat([x4_up2, x3], dim=1)
        x_zz3 = self.enhance_Conv1(x3_add)
# -----------------第三个enhance module----------------------
        x_out2 = self.up2(x_out1, x_zz3)
# -----------------第二个enhance module----------------------
        x4_up4 = self.upz1(x4_up2)
        x3_up2 = self.upz1(x3)
        x2_add = torch.cat([x4_up4, x3_up2, x2], dim=1)
        x_zz2 = self.enhance_Conv2(x2_add)
# -----------------第二个enhance module----------------------
        x_out3 = self.up3(x_out2, x_zz2)
# -----------------第一个enhance module----------------------
        x4_up8 = self.upz1(x4_up4)
        x3_up4 = self.upz1(x3_up2)
        x2_up2 = self.upz1(x2)
        x1_add = torch.cat([x4_up8, x3_up4, x2_up2, x1], dim=1)
        x_zz1 = self.enhance_Conv3(x1_add)
# -----------------第一个enhance module----------------------
        x_out4 = self.up4(x_out3, x_zz1)
        logits = self.out_conv(x_out4)

        return {"out": logits}


# ---------------------方案7.5（Skip_enhance）--------------------------------
class UNet_Skip_enhance_two(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet_Skip_enhance_two, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)
        # self.CARAFE1 = CARAFE(base_c * 2, scale=2)
        # self.CARAFE2 = CARAFE(base_c * 4, scale=4)
        # self.CARAFE3 = CARAFE(base_c * 8, scale=8)
        # 跳跃加强模块1中的上采样
        self.upz1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upz2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upz3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.enhance_Conv1 = Conv(base_c * 12, base_c * 4, k=3, s=1, p=1, g=1, act=nn.ReLU())
        self.enhance_Conv2 = Conv(base_c * 14, base_c * 2, k=3, s=1, p=1, g=1, act=nn.ReLU())
        self.enhance_Conv3 = Conv(base_c * 15, base_c * 1, k=3, s=1, p=1, g=1, act=nn.ReLU())

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x_out1 = self.up1(x5, x4)
# -----------------第三个enhance module----------------------
        x4_up2 = self.upz1(x4)
        # x3_add = torch.cat([x4_up2, x3], dim=1)
        # x_zz3 = self.enhance_Conv1(x3_add)
# -----------------第三个enhance module----------------------
        x_out2 = self.up2(x_out1, x3)
# -----------------第二个enhance module----------------------
        x4_up4 = self.upz1(x4_up2)
        x3_up2 = self.upz1(x3)
        x2_add = torch.cat([x4_up4, x3_up2, x2], dim=1)
        x_zz2 = self.enhance_Conv2(x2_add)
# -----------------第二个enhance module----------------------
        x_out3 = self.up3(x_out2, x_zz2)
# -----------------第一个enhance module----------------------
        x4_up8 = self.upz1(x4_up4)
        x3_up4 = self.upz1(x3_up2)
        x2_up2 = self.upz1(x2)
        x1_add = torch.cat([x4_up8, x3_up4, x2_up2, x1], dim=1)
        x_zz1 = self.enhance_Conv3(x1_add)
# -----------------第一个enhance module----------------------
        x_out4 = self.up4(x_out3, x_zz1)
        logits = self.out_conv(x_out4)

        return {"out": logits}


# ---------------------方案7.9（Skip_enhance）--------------------------------
class UNet_Skip_enhance_one(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet_Skip_enhance_one, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)
        # self.CARAFE1 = CARAFE(base_c * 2, scale=2)
        # self.CARAFE2 = CARAFE(base_c * 4, scale=4)
        # self.CARAFE3 = CARAFE(base_c * 8, scale=8)
        # 跳跃加强模块1中的上采样
        self.upz1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upz2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upz3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.enhance_Conv1 = Conv(base_c * 12, base_c * 4, k=3, s=1, p=1, g=1, act=nn.ReLU())
        self.enhance_Conv2 = Conv(base_c * 14, base_c * 2, k=3, s=1, p=1, g=1, act=nn.ReLU())
        self.enhance_Conv3 = Conv(base_c * 15, base_c * 1, k=3, s=1, p=1, g=1, act=nn.ReLU())


    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x_out1 = self.up1(x5, x4)
# -----------------第三个enhance module----------------------
        x4_up2 = self.upz1(x4)
        # x3_add = torch.cat([x4_up2, x3], dim=1)
        # x_zz3 = self.enhance_Conv1(x3_add)
# -----------------第三个enhance module----------------------
        x_out2 = self.up2(x_out1, x3)
# -----------------第二个enhance module----------------------
        x4_up4 = self.upz1(x4_up2)
        x3_up2 = self.upz1(x3)
        # x2_add = torch.cat([x4_up4, x3_up2, x2], dim=1)
        # x_zz2 = self.enhance_Conv2(x2_add)
# -----------------第二个enhance module----------------------
        x_out3 = self.up3(x_out2, x2)
# -----------------第一个enhance module----------------------
        x4_up8 = self.upz1(x4_up4)
        x3_up4 = self.upz1(x3_up2)
        x2_up2 = self.upz1(x2)
        x1_add = torch.cat([x4_up8, x3_up4, x2_up2, x1], dim=1)
        x_zz1 = self.enhance_Conv3(x1_add)
# -----------------第一个enhance module----------------------
        x_out4 = self.up4(x_out3, x_zz1)
        logits = self.out_conv(x_out4)

        return {"out": logits}


# ---------------------方案7.99（Skip_enhance）--------------------------------
class UNet_Skip_enhance_one_CBAM(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet_Skip_enhance_one_CBAM, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)
        # self.CARAFE1 = CARAFE(base_c * 2, scale=2)
        # self.CARAFE2 = CARAFE(base_c * 4, scale=4)
        # self.CARAFE3 = CARAFE(base_c * 8, scale=8)
        # 跳跃加强模块1中的上采样
        self.upz1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upz2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upz3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.enhance_Conv1 = Conv(base_c * 12, base_c * 4, k=3, s=1, p=1, g=1, act=nn.ReLU())
        self.enhance_Conv2 = Conv(base_c * 14, base_c * 2, k=3, s=1, p=1, g=1, act=nn.ReLU())
        self.enhance_Conv3 = Conv(base_c * 15, base_c * 1, k=3, s=1, p=1, g=1, act=nn.ReLU())
        self.cbam1 = CBAM(base_c)
        self.cbam2 = CBAM(base_c * 2)
        self.cbam3 = CBAM(base_c * 4)
        self.cbam4 = CBAM(base_c * 8)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x4 = self.cbam4(x4)                                   # CBAM4
        x_out1 = self.up1(x5, x4)
        # -----------------第三个enhance module----------------------
        x4_up2 = self.upz1(x4)
        # x3_add = torch.cat([x4_up2, x3], dim=1)
        # x_zz3 = self.enhance_Conv1(x3_add)
        # -----------------第三个enhance module----------------------
        x3 = self.cbam3(x3)                             # CBAM3
        x_out2 = self.up2(x_out1, x3)
        # -----------------第二个enhance module----------------------
        x4_up4 = self.upz1(x4_up2)
        x3_up2 = self.upz1(x3)
        # x2_add = torch.cat([x4_up4, x3_up2, x2], dim=1)
        # x_zz2 = self.enhance_Conv2(x2_add)
        # -----------------第二个enhance module----------------------
        x2 = self.cbam2(x2)                             # CBAM2
        x_out3 = self.up3(x_out2, x2)
        # -----------------第一个enhance module----------------------
        x4_up8 = self.upz1(x4_up4)
        x3_up4 = self.upz1(x3_up2)
        x2_up2 = self.upz1(x2)
        x1_add = torch.cat([x4_up8, x3_up4, x2_up2, x1], dim=1)
        x_zz1 = self.enhance_Conv3(x1_add)
        # -----------------第一个enhance module----------------------
        x_zz1 = self.cbam1(x_zz1)                             # CBAM1
        x_out4 = self.up4(x_out3, x_zz1)
        logits = self.out_conv(x_out4)

        return {"out": logits}


# ---------------------方案8（Skip_enhance_CBAM）---------------------------------------------
class UNet_Skip_enhance_CBAM(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet_Skip_enhance_CBAM, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)
        # self.CARAFE1 = CARAFE(base_c * 2, scale=2)
        # self.CARAFE2 = CARAFE(base_c * 4, scale=4)
        # self.CARAFE3 = CARAFE(base_c * 8, scale=8)
        # 跳跃加强模块1中的上采样
        self.upz1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upz2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upz3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.enhance_Conv1 = Conv(base_c * 12, base_c * 4, k=3, s=1, p=1, g=1, act=nn.ReLU())
        self.enhance_Conv2 = Conv(base_c * 14, base_c * 2, k=3, s=1, p=1, g=1, act=nn.ReLU())
        self.enhance_Conv3 = Conv(base_c * 15, base_c * 1, k=3, s=1, p=1, g=1, act=nn.ReLU())
        self.cbam1 = CBAM(base_c)
        self.cbam2 = CBAM(base_c * 2)
        self.cbam3 = CBAM(base_c * 4)
        self.cbam4 = CBAM(base_c * 8)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x4 = self.cbam4(x4)                                   # CBAM4
        x_out1 = self.up1(x5, x4)
# -----------------第三个enhance module----------------------
        x4_up2 = self.upz1(x4)
        x3_add = torch.cat([x4_up2, x3], dim=1)
        x_zz3 = self.enhance_Conv1(x3_add)
        x_zz3 = self.cbam3(x_zz3)                             # CBAM3
# -----------------第三个enhance module----------------------
        x_out2 = self.up2(x_out1, x_zz3)
# -----------------第二个enhance module----------------------
        x4_up4 = self.upz1(x4_up2)
        x3_up2 = self.upz1(x3)
        x2_add = torch.cat([x4_up4, x3_up2, x2], dim=1)
        x_zz2 = self.enhance_Conv2(x2_add)
        x_zz2 = self.cbam2(x_zz2)                             # CBAM2
# -----------------第二个enhance module----------------------
        x_out3 = self.up3(x_out2, x_zz2)
# -----------------第一个enhance module----------------------
        x4_up8 = self.upz1(x4_up4)
        x3_up4 = self.upz1(x3_up2)
        x2_up2 = self.upz1(x2)
        x1_add = torch.cat([x4_up8, x3_up4, x2_up2, x1], dim=1)
        x_zz1 = self.enhance_Conv3(x1_add)
        x_zz1 = self.cbam1(x_zz1)                             # CBAM1
# -----------------第一个enhance module----------------------
        x_out4 = self.up4(x_out3, x_zz1)
        logits = self.out_conv(x_out4)

        return {"out": logits}




# ---------------------方案18（UNet_Skip_enhance_CBAM_FasterNetBlock(*************FEC-Net*************)）---------------------------------------------------

class DoubleConv_FasterNetBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv_FasterNetBlock, self).__init__(
            # nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            FasterNetBlock(in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # nn.GELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            # nn.GELU()
        )

class Down7(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down7, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv_FasterNetBlock(in_channels, out_channels)
        )

class Up7(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up7, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)              # 张震
            self.conv = DoubleConv_FasterNetBlock(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv_FasterNetBlock(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class OutConv7(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv7, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

class UNet_Skip_enhance_CBAM_FasterNetBlock(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet_Skip_enhance_CBAM_FasterNetBlock, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down7(base_c, base_c * 2)
        self.down2 = Down7(base_c * 2, base_c * 4)
        self.down3 = Down7(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down7(base_c * 8, base_c * 16 // factor)
        self.up1 = Up7(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up7(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up7(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up7(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv7(base_c, num_classes)
        # 跳跃加强模块1中的上采样
        self.upz1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upz2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upz3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.enhance_Conv1 = Conv(base_c * 12, base_c * 4, k=3, s=1, p=1, g=1, act=nn.ReLU())
        self.enhance_Conv2 = Conv(base_c * 14, base_c * 2, k=3, s=1, p=1, g=1, act=nn.ReLU())
        self.enhance_Conv3 = Conv(base_c * 15, base_c * 1, k=3, s=1, p=1, g=1, act=nn.ReLU())
        self.cbam1 = CBAM(base_c)
        self.cbam2 = CBAM(base_c * 2)
        self.cbam3 = CBAM(base_c * 4)
        self.cbam4 = CBAM(base_c * 8)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x4 = self.cbam4(x4)                                   # CBAM4
        x_out1 = self.up1(x5, x4)
# -----------------第三个enhance module----------------------
        x4_up2 = self.upz1(x4)
        x3_add = torch.cat([x4_up2, x3], dim=1)
        x_zz3 = self.enhance_Conv1(x3_add)
        x_zz3 = self.cbam3(x_zz3)                             # CBAM3
# -----------------第三个enhance module----------------------
        x_out2 = self.up2(x_out1, x_zz3)
# -----------------第二个enhance module----------------------
        x4_up4 = self.upz1(x4_up2)
        x3_up2 = self.upz1(x3)
        x2_add = torch.cat([x4_up4, x3_up2, x2], dim=1)
        x_zz2 = self.enhance_Conv2(x2_add)
        x_zz2 = self.cbam2(x_zz2)                             # CBAM2
# -----------------第二个enhance module----------------------
        x_out3 = self.up3(x_out2, x_zz2)
# -----------------第一个enhance module----------------------
        x4_up8 = self.upz1(x4_up4)
        x3_up4 = self.upz1(x3_up2)
        x2_up2 = self.upz1(x2)
        x1_add = torch.cat([x4_up8, x3_up4, x2_up2, x1], dim=1)
        x_zz1 = self.enhance_Conv3(x1_add)
        x_zz1 = self.cbam1(x_zz1)                             # CBAM1
# -----------------第一个enhance module----------------------
        x_out4 = self.up4(x_out3, x_zz1)
        logits = self.out_conv(x_out4)

        return {"out": logits}




# ---------------------------------------24unet_FasterNetBlock--------------------------------
class unet_FasterNetBlock(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(unet_FasterNetBlock, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down7(base_c, base_c * 2)
        self.down2 = Down7(base_c * 2, base_c * 4)
        self.down3 = Down7(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down7(base_c * 8, base_c * 16 // factor)
        self.up1 = Up7(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up7(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up7(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up7(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv7(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return {"out": logits}


# ---------------------------------------25unet_Skip_enhance_FasterNetBlock--------------------------------
class UNet_Skip_enhance_FasterNetBlock(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet_Skip_enhance_FasterNetBlock, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down7(base_c, base_c * 2)
        self.down2 = Down7(base_c * 2, base_c * 4)
        self.down3 = Down7(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down7(base_c * 8, base_c * 16 // factor)
        self.up1 = Up7(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up7(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up7(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up7(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv7(base_c, num_classes)
        # self.CARAFE1 = CARAFE(base_c * 2, scale=2)
        # self.CARAFE2 = CARAFE(base_c * 4, scale=4)
        # self.CARAFE3 = CARAFE(base_c * 8, scale=8)
        # 跳跃加强模块1中的上采样
        self.upz1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upz2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upz3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.enhance_Conv1 = Conv(base_c * 12, base_c * 4, k=3, s=1, p=1, g=1, act=nn.ReLU())
        self.enhance_Conv2 = Conv(base_c * 14, base_c * 2, k=3, s=1, p=1, g=1, act=nn.ReLU())
        self.enhance_Conv3 = Conv(base_c * 15, base_c * 1, k=3, s=1, p=1, g=1, act=nn.ReLU())

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x_out1 = self.up1(x5, x4)
# -----------------第三个enhance module----------------------
        x4_up2 = self.upz1(x4)
        x3_add = torch.cat([x4_up2, x3], dim=1)
        x_zz3 = self.enhance_Conv1(x3_add)
# -----------------第三个enhance module----------------------
        x_out2 = self.up2(x_out1, x_zz3)
# -----------------第二个enhance module----------------------
        x4_up4 = self.upz1(x4_up2)
        x3_up2 = self.upz1(x3)
        x2_add = torch.cat([x4_up4, x3_up2, x2], dim=1)
        x_zz2 = self.enhance_Conv2(x2_add)
# -----------------第二个enhance module----------------------
        x_out3 = self.up3(x_out2, x_zz2)
# -----------------第一个enhance module----------------------
        x4_up8 = self.upz1(x4_up4)
        x3_up4 = self.upz1(x3_up2)
        x2_up2 = self.upz1(x2)
        x1_add = torch.cat([x4_up8, x3_up4, x2_up2, x1], dim=1)
        x_zz1 = self.enhance_Conv3(x1_add)
# -----------------第一个enhance module----------------------
        x_out4 = self.up4(x_out3, x_zz1)
        logits = self.out_conv(x_out4)

        return {"out": logits}


# ---------------------------------------26UNet_CBAM_FasterNetBlock--------------------------------
class UNet_CBAM_FasterNetBlock(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet_CBAM_FasterNetBlock, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down7(base_c, base_c * 2)
        self.down2 = Down7(base_c * 2, base_c * 4)
        self.down3 = Down7(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down7(base_c * 8, base_c * 16 // factor)
        self.up1 = Up7(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up7(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up7(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up7(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv7(base_c, num_classes)
        self.cbam1 = CBAM(base_c)
        self.cbam2 = CBAM(base_c * 2)
        self.cbam3 = CBAM(base_c * 4)
        self.cbam4 = CBAM(base_c * 8)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x4 = self.cbam4(x4)    # zz
        x = self.up1(x5, x4)
        x3 = self.cbam3(x3)    # zz
        x = self.up2(x, x3)
        x2 = self.cbam2(x2)    # zz
        x = self.up3(x, x2)
        x1 = self.cbam1(x1)  # zz
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return {"out": logits}
# ---------------------------------------27 --------------------------------






#--------------------------------------------改进的部分------------------------------------------
# -------------------------------------------CBAM----------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
        max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1*h*w
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        # 2*h*w
        x = self.conv(x)
        # 1*h*w
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, c1, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(c1, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x) * x
        # c*h*w
        # c*h*w * 1*h*w
        out = self.spatial_attention(out) * out
        return out


# ---------------------------------------------PConv----------------------------------------------------------------------
class PConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 kernel_size=3,
                 n_div: int = 4,
                 forward: str = 'split_cat'):
        super(PConv2d, self).__init__()
        assert in_channels > 4, "in_channels should > 4, but got {} instead.".format(in_channels)
        self.dim_conv = in_channels // n_div
        self.dim_untouched = in_channels - self.dim_conv

        self.conv = nn.Conv2d(in_channels=self.dim_conv,
                              out_channels=self.dim_conv,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=(kernel_size - 1) // 2,
                              bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing

        elif forward == 'split_cat':
            self.forward = self.forward_split_cat

        else:
            raise NotImplementedError("forward method: {} is not implemented.".format(forward))

    def forward_slicing(self, x: Tensor) -> Tensor:
        x[:, :self.dim_conv, :, :] = self.conv(x[:, :self.dim_conv, :, :])

        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.conv(x1)
        x = torch.cat((x1, x2), dim=1)

        return x


class ConvBNLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 bias: bool = False,
                 act: str = 'ReLU'):
        super(ConvBNLayer, self).__init__()
        assert act in ('ReLU', 'GELU')
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride,
                              padding,
                              dilation,
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = getattr(nn, act)()

    def _fuse_bn_tensor(self) -> None:
        kernel = self.conv.weight
        bias = self.conv.bias if hasattr(self.conv, 'bias') and self.conv.bias is not None else 0
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        gamma = self.bn.weight
        beta = self.bn.bias
        eps = self.bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        self.conv.weight.data = kernel * t
        self.conv.bias = nn.Parameter(beta - (running_mean - bias) * gamma / std)
        self.bn = nn.Identity()
        return self.conv.weight.data, self.conv.bias.data

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class FasterNetBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 inner_channels: int = None,
                 kernel_size: int = 3,
                 bias=False,
                 act: str = 'ReLU',
                 n_div: int = 4,
                 forward: str = 'split_cat',
                 ):
        super(FasterNetBlock, self).__init__()
        inner_channels = inner_channels or in_channels * 2
        self.conv1 = PConv2d(in_channels,
                             kernel_size,
                             n_div,
                             forward)
        self.conv2 = ConvBNLayer(in_channels,
                                 inner_channels,
                                 bias=bias,
                                 act=act)
        self.conv3 = nn.Conv2d(inner_channels,
                               in_channels,
                               kernel_size=1,
                               stride=1,
                               bias=True)

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)

        return x + y








