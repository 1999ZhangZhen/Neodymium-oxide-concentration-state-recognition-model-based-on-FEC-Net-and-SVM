import math
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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

class Up_CARAFE(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up_CARAFE, self).__init__()
        if bilinear:
            # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)              # 张震
            self.UP_CARAFE = CARAFE(c = in_channels // 2)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.UP_CARAFE(x1)
        # x1 = self.up(x1)
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

# ---------------------方案3--------------------
class UNet_CARAFE(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet_CARAFE, self).__init__()
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

# ---------------------方案4--------------------------------
class UNet_SPPF(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet_SPPF, self).__init__()
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
        self.upz1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upz2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upz3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.SPPFConv = Conv(base_c * 15, base_c, k=3, s=1, p=1, g=1, act=nn.ReLU())


    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
# -----------------SPPF----------------------
        x_2 = self.upz1(x2)
        x_3 = self.upz2(x3)
        x_4 = self.upz3(x4)
        x_add = torch.cat([x_4, x_3, x_2, x1], dim=1)
        x_zz = self.SPPFConv(x_add)
        x = self.up4(x, x_zz)
        logits = self.out_conv(x)

        return {"out": logits}


# ---------------------方案5--------------------------------
class UNet_CBAM_SPPF(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet_CBAM_SPPF, self).__init__()
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
        self.upz1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upz2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upz3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.SPPFConv = Conv(base_c * 15, base_c, k=3, s=1, p=1, g=1, act=nn.ReLU())


    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        #---------------sppf--------------------
        xz_4 = self.upz3(x4)
        xz_3 = self.upz2(x3)
        xz_2 = self.upz1(x2)
        x4 = self.cbam4(x4)    # zz
        x = self.up1(x5, x4)
        x3 = self.cbam3(x3)    # zz
        x = self.up2(x, x3)
        x2 = self.cbam2(x2)    # zz
        x = self.up3(x, x2)
        x_add = torch.cat([xz_4, xz_3, xz_2, x1], dim=1)
        x_adds = self.SPPFConv(x_add)

        x_adds = self.cbam1(x_adds)  # zz
        x = self.up4(x, x_adds)
        logits = self.out_conv(x)

        return {"out": logits}

# ---------------------方案6--------------------------------
class UNet_CBAM_RESPPF(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet_CBAM_RESPPF, self).__init__()
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
        self.upz1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upz2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upz3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upz4 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.SPPFConv = Conv(base_c * 15, base_c, k=3, s=1, p=1, g=1, act=nn.ReLU())
        self.convzz = DoubleConv(base_c * 2, base_c)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        #---------------sppf--------------------
        xz_4 = self.upz3(x4)
        xz_3 = self.upz2(x3)
        xz_2 = self.upz1(x2)
        xzz_1 = self.upz4(x5)
        x4 = self.cbam4(x4)    # zz
        xa = self.up1(x5, x4)
        xzz_2 = self.upz3(xa)
        x3 = self.cbam3(x3)    # zz
        xb = self.up2(xa, x3)
        xzz_3 = self.upz2(xb)
        x2 = self.cbam2(x2)    # zz
        xc = self.up3(xb, x2)
        xzz_4 = self.upz1(xc)
        x_add = torch.cat([xz_4, xz_3, xz_2, x1], dim=1)
        x_addzz = torch.cat([xzz_1, xzz_2, xzz_3, xzz_4], dim=1)
        x_adds = self.SPPFConv(x_add)
        x_addzzs = self.SPPFConv(x_addzz)
        x_adds = self.cbam1(x_adds)  # zz
        add = torch.cat([x_adds, x_addzzs], dim=1)
        x = self.convzz(add)
        # xd = self.up4(xc, x_adds)
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


# ---------------------方案8（Skip_enhance_CBAM）------------------------------------------------------------------------------------------------------
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


# ---------------------方案9（Skip_enhance）--------------------------------
class UNet_Skip_enhance_CARAFE(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet_Skip_enhance_CARAFE, self).__init__()
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
        self.CARAFE1 = CARAFE(base_c * 2, scale=2)
        self.CARAFE2 = CARAFE(base_c * 4, scale=2)
        self.CARAFE3 = CARAFE(base_c * 8, scale=2)
        # 跳跃加强模块1中的上采样
        # self.upz1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.upz2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        # self.upz3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
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
        x4_up2 = self.CARAFE3(x4)
        x3_add = torch.cat([x4_up2, x3], dim=1)
        x_zz3 = self.enhance_Conv1(x3_add)
# -----------------第三个enhance module----------------------
        x_out2 = self.up2(x_out1, x_zz3)
# -----------------第二个enhance module----------------------
        x4_up4 = self.CARAFE3(x4_up2)
        x3_up2 = self.CARAFE2(x3)
        x2_add = torch.cat([x4_up4, x3_up2, x2], dim=1)
        x_zz2 = self.enhance_Conv2(x2_add)
# -----------------第二个enhance module----------------------
        x_out3 = self.up3(x_out2, x_zz2)
# -----------------第一个enhance module----------------------
        x4_up8 = self.CARAFE3(x4_up4)
        x3_up4 = self.CARAFE2(x3_up2)
        x2_up2 = self.CARAFE1(x2)
        x1_add = torch.cat([x4_up8, x3_up4, x2_up2, x1], dim=1)
        x_zz1 = self.enhance_Conv3(x1_add)
# -----------------第一个enhance module----------------------
        x_out4 = self.up4(x_out3, x_zz1)
        logits = self.out_conv(x_out4)

        return {"out": logits}



# ---------------------方案10（UNet_Skip_enhance_CBAM_coordconv）--------------------------------
class UNet_Skip_enhance_CBAM_coordconv(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet_Skip_enhance_CBAM_coordconv, self).__init__()
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
        self.out_coordconv = CoordConv(base_c, num_classes)
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
        logits = self.out_coordconv(x_out4)

        return {"out": logits}






# ---------------------方案11（UNet_Skip_enhance_CBAM_dsconv）---------------------------------------------------

class DoubleConv_dsConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv_dsConv, self).__init__(
            # nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            DSConv2D(in_channels, mid_channels, 3),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # nn.GELU(),
            # nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            DSConv2D(mid_channels, out_channels, 3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            # nn.GELU()
        )

class Down1(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down1, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv_dsConv(in_channels, out_channels)
        )

class Up1(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up1, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)              # 张震
            self.conv = DoubleConv_dsConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv_dsConv(in_channels, out_channels)

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

class OutConv1(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv1, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class UNet_Skip_enhance_CBAM_dsconv(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet_Skip_enhance_CBAM_dsconv, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down1(base_c, base_c * 2)
        self.down2 = Down1(base_c * 2, base_c * 4)
        self.down3 = Down1(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down1(base_c * 8, base_c * 16 // factor)
        self.up1 = Up1(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up1(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up1(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up1(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv1(base_c, num_classes)
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


# ---------------------方案12（UNet_Skip_enhance_CBAM_saconv）---------------------------------------------------

class DoubleConv_saConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv_saConv, self).__init__(
            # nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            SAConv2d(in_channels, mid_channels, 3),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # nn.GELU(),
            # nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            SAConv2d(mid_channels, out_channels, 3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            # nn.GELU()
        )

class Down2(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down2, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv_saConv(in_channels, out_channels)
        )

class Up2(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up2, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)              # 张震
            self.conv = DoubleConv_saConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv_saConv(in_channels, out_channels)

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

class OutConv2(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv2, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class UNet_Skip_enhance_CBAM_saconv(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet_Skip_enhance_CBAM_saconv, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down2(base_c, base_c * 2)
        self.down2 = Down2(base_c * 2, base_c * 4)
        self.down3 = Down2(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down2(base_c * 8, base_c * 16 // factor)
        self.up1 = Up2(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up2(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up2(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up2(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv2(base_c, num_classes)
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




# ---------------------方案13（UNet_Skip_enhance_CBAM_CARAFE）------------------------------------------------------------------------------------------------------


class DoubleConv_CARAFE(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv_CARAFE, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            # nn.GELU()
        )


class Down3(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down3, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv_CARAFE(in_channels, out_channels)
        )


class Up_CARAFE(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up_CARAFE, self).__init__()
        if bilinear:
            # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)              # 张震
            self.UP_CARAFE = CARAFE(in_channels // 2)
            self.conv = DoubleConv_CARAFE(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv_CARAFE(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.UP_CARAFE(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class OutConv3(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv3, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

class UNet_Skip_enhance_CBAM_CARAFE(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet_Skip_enhance_CBAM_CARAFE, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down3(base_c, base_c * 2)
        self.down2 = Down3(base_c * 2, base_c * 4)
        self.down3 = Down3(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down3(base_c * 8, base_c * 16 // factor)
        self.up1 = Up_CARAFE(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up_CARAFE(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up_CARAFE(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up_CARAFE(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv3(base_c, num_classes)
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





# ---------------------方案14（UNet_Skip_enhance_CBAM_DCNv2）---------------------------------------------------

class DoubleConv_DCN(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv_DCN, self).__init__(
            # nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            DCNv2(in_channels, mid_channels, kernel_size=3),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # DCNv2(mid_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            # nn.GELU()
        )


class Down4(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down4, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv_DCN(in_channels, out_channels)
        )


class Up4(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up4, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)              # 张震
            self.conv = DoubleConv_DCN(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv_DCN(in_channels, out_channels)

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

class OutConv4(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv4, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class UNet_Skip_enhance_CBAM_DCN(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet_Skip_enhance_CBAM_DCN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down4(base_c, base_c * 2)
        self.down2 = Down4(base_c * 2, base_c * 4)
        self.down3 = Down4(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down4(base_c * 8, base_c * 16 // factor)
        self.up1 = Up4(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up4(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up4(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up4(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv4(base_c, num_classes)
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


# ---------------------方案15（UNet_Skip_enhance_CBAM_coordconv）---------------------------------------------------

class DoubleConv_2coordconv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv_2coordconv, self).__init__(
            # nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            CoordConv(in_channels, mid_channels, kernel_size=3),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # nn.GELU(),
            # nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            CoordConv(mid_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            # nn.GELU()
        )


class Down5(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down5, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv_2coordconv(in_channels, out_channels)
        )


class Up5(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up5, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)              # 张震
            self.conv = DoubleConv_2coordconv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv_2coordconv(in_channels, out_channels)

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

class OutConv5(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv5, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class UNet_Skip_enhance_CBAM_2coordconv(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet_Skip_enhance_CBAM_2coordconv, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down5(base_c, base_c * 2)
        self.down2 = Down5(base_c * 2, base_c * 4)
        self.down3 = Down5(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down5(base_c * 8, base_c * 16 // factor)
        self.up1 = Up5(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up5(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up5(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up5(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv5(base_c, num_classes)
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




# ---------------------方案17（UNet_Skip_enhance_CBAM_C2f_ODConv2d）---------------------------------------------------

class DoubleConv_C2f(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv_C2f, self).__init__(
            # nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            C2f(in_channels, mid_channels),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # ODConv2d(mid_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            # nn.GELU()
        )


class Down6(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down6, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv_C2f(in_channels, out_channels)
        )


class Up6(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up6, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)              # 张震
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

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

class OutConv6(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv6, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class UNet_Skip_enhance_CBAM_C2f(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet_Skip_enhance_CBAM_C2f, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down6(base_c, base_c * 2)
        self.down2 = Down6(base_c * 2, base_c * 4)
        self.down3 = Down6(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down6(base_c * 8, base_c * 16 // factor)
        self.up1 = Up6(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up6(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up6(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up6(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv6(base_c, num_classes)
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




# ---------------------方案18（UNet_Skip_enhance_CBAM_FasterNetBlock）---------------------------------------------------

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


# ---------------------方案19（UNet_Skip_enhance_CBAM_FasterNetBlock_bianma）---------------------------------------------------

class DoubleConv_FasterNetBlock_bianma(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv_FasterNetBlock_bianma, self).__init__(
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


class Down8(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down8, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv_FasterNetBlock_bianma(in_channels, out_channels)
        )


class Up8(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up8, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)              # 张震
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

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

class OutConv8(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv8, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class UNet_Skip_enhance_CBAM_FasterNetBlock_bianma(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet_Skip_enhance_CBAM_FasterNetBlock_bianma, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down8(base_c, base_c * 2)
        self.down2 = Down8(base_c * 2, base_c * 4)
        self.down3 = Down8(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down8(base_c * 8, base_c * 16 // factor)
        self.up1 = Up8(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up8(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up8(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up8(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv8(base_c, num_classes)
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


# ---------------------方案20（UNet_Skip_enhance_CBAM_FasterNetBlock_jeima）---------------------------------------------------

class DoubleConv_FasterNetBlock_jeima(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv_FasterNetBlock_jeima, self).__init__(
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


class Down9(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down9, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up9(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up9, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)              # 张震
            self.conv = DoubleConv_FasterNetBlock_jeima(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv_FasterNetBlock_jeima(in_channels, out_channels)

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

class OutConv9(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv9, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class UNet_Skip_enhance_CBAM_FasterNetBlock_jeima(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet_Skip_enhance_CBAM_FasterNetBlock_jeima, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down9(base_c, base_c * 2)
        self.down2 = Down9(base_c * 2, base_c * 4)
        self.down3 = Down9(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down9(base_c * 8, base_c * 16 // factor)
        self.up1 = Up9(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up9(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up9(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up9(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv9(base_c, num_classes)
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



# ---------------------方案21（UNet_Skip_enhance_CBAM_MobileNext）---------------------------------------------------

class DoubleConv_MobileNext(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv_MobileNext, self).__init__(
            # nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            SGBlock(in_channels, mid_channels, 1, 4),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            # nn.GELU()
        )


class Down10(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down10, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv_MobileNext(in_channels, out_channels)
        )


class Up10(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up10, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)              # 张震
            self.conv = DoubleConv_MobileNext(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv_MobileNext(in_channels, out_channels)

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

class OutConv10(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv10, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class UNet_Skip_enhance_CBAM_MobileNext(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet_Skip_enhance_CBAM_MobileNext, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down10(base_c, base_c * 2)
        self.down2 = Down10(base_c * 2, base_c * 4)
        self.down3 = Down10(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down10(base_c * 8, base_c * 16 // factor)
        self.up1 = Up10(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up10(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up10(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up10(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv10(base_c, num_classes)
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


# ---------------------方案22（UNet_Skip_enhance_CBAM_MobileNext_bianma）---------------------------------------------------

class DoubleConv_FasterNetBlock_MobileNext_bianma(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv_FasterNetBlock_MobileNext_bianma, self).__init__(
            # nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            SGBlock(in_channels, mid_channels, 1, 6),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            # nn.GELU()
        )


class Down11(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down11, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv_FasterNetBlock_MobileNext_bianma(in_channels, out_channels)
        )


class Up11(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up11, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)              # 张震
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

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

class OutConv11(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv11, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class UNet_Skip_enhance_CBAM_MobileNext_bianma(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet_Skip_enhance_CBAM_MobileNext_bianma, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down11(base_c, base_c * 2)
        self.down2 = Down11(base_c * 2, base_c * 4)
        self.down3 = Down11(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down11(base_c * 8, base_c * 16 // factor)
        self.up1 = Up11(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up11(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up11(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up11(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv11(base_c, num_classes)
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


# ---------------------方案23（UNet_Skip_enhance_CBAM_MobileNext_bianma）---------------------------------------------------

class DoubleConv_FasterNetBlock_MobileNext_jeima(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv_FasterNetBlock_MobileNext_jeima, self).__init__(
            # nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            SGBlock(in_channels, mid_channels, 1, 6),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            # nn.GELU()
        )


class Down12(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down12, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up12(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up12, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)              # 张震
            self.conv = DoubleConv_FasterNetBlock_MobileNext_jeima(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv_FasterNetBlock_MobileNext_jeima(in_channels, out_channels)

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

class OutConv12(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv12, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class UNet_Skip_enhance_CBAM_MobileNext_jeima(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet_Skip_enhance_CBAM_MobileNext_jeima, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down12(base_c, base_c * 2)
        self.down2 = Down12(base_c * 2, base_c * 4)
        self.down3 = Down12(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down12(base_c * 8, base_c * 16 // factor)
        self.up1 = Up12(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up12(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up12(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up12(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv12(base_c, num_classes)
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

# -----------------------------------------------CARAFE----------------------------------------------------
def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
class CARAFE(nn.Module):
    def __init__(self, c, k_enc=3, k_up=5, c_mid=64, scale=2):
        """ The unofficial implementation of the CARAFE module.
        The details are in "https://arxiv.org/abs/1905.02188".
        Args:
            c: The channel number of the input and the output.
            c_mid: The channel number after compression.
            scale: The expected upsample scale.
            k_up: The size of the reassembly kernel.
            k_enc: The kernel size of the encoder.
        Returns:
            X: The upsampled feature map.
        """
        super(CARAFE, self).__init__()
        self.scale = scale

        self.comp = Conv(c, c_mid, act=nn.ReLU())
        self.enc = Conv(c_mid, (scale * k_up) ** 2, k=k_enc, act=False)
        self.pix_shf = nn.PixelShuffle(scale)

        self.upsmp = nn.Upsample(scale_factor=scale, mode='nearest')
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale,
                                padding=k_up // 2 * scale)

    def forward(self, X):
        b, c, h, w = X.size()
        h_, w_ = h * self.scale, w * self.scale

        W = self.comp(X)  # b * m * h * w
        W = self.enc(W)  # b * 100 * h * w
        W = self.pix_shf(W)  # b * 25 * h_ * w_
        W = torch.softmax(W, dim=1)  # b * 25 * h_ * w_

        X = self.upsmp(X)  # b * c * h_ * w_
        X = self.unfold(X)  # b * 25c * h_ * w_
        X = X.view(b, c, -1, h_, w_)  # b * 25 * c * h_ * w_

        X = torch.einsum('bkhw,bckhw->bchw', [W, X])  # b * c * h_ * w_
        return X


# -----------------------------------------------coordconv坐标卷积----------------------------------------------------
class AddCoords(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, with_r=False):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_channels += 2
        if with_r:
            in_channels += 1
        self.conv = Conv(in_channels, out_channels, k=kernel_size, s=stride)

    def forward(self, x):
        x = self.addcoords(x)
        x = self.conv(x)
        return x


# -----------------------------------------------SAConv----------------------------------------------------
class ConvAWS2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.register_buffer('weight_gamma', torch.ones(self.out_channels, 1, 1, 1))
        self.register_buffer('weight_beta', torch.zeros(self.out_channels, 1, 1, 1))

    def _get_weight(self, weight):
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                                            keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = torch.sqrt(weight.view(weight.size(0), -1).var(dim=1) + 1e-5).view(-1, 1, 1, 1)
        weight = weight / std
        weight = self.weight_gamma * weight + self.weight_beta
        return weight

    def forward(self, x):
        weight = self._get_weight(self.weight)
        return super()._conv_forward(x, weight, None)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self.weight_gamma.data.fill_(-1)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        if self.weight_gamma.data.mean() > 0:
            return
        weight = self.weight.data
        weight_mean = weight.data.mean(dim=1, keepdim=True).mean(dim=2,
                                                                 keepdim=True).mean(dim=3, keepdim=True)
        self.weight_beta.data.copy_(weight_mean)
        std = torch.sqrt(weight.view(weight.size(0), -1).var(dim=1) + 1e-5).view(-1, 1, 1, 1)
        self.weight_gamma.data.copy_(std)


class SAConv2d(ConvAWS2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 s=1,
                 p=None,
                 g=1,
                 d=1,
                 act=True,
                 bias=True):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=s,
            padding=autopad(kernel_size, p, d),
            dilation=d,
            groups=g,
            bias=bias)
        self.switch = torch.nn.Conv2d(
            self.in_channels,
            1,
            kernel_size=1,
            stride=s,
            bias=True)
        self.switch.weight.data.fill_(0)
        self.switch.bias.data.fill_(1)
        self.weight_diff = torch.nn.Parameter(torch.Tensor(self.weight.size()))
        self.weight_diff.data.zero_()
        self.pre_context = torch.nn.Conv2d(
            self.in_channels,
            self.in_channels,
            kernel_size=1,
            bias=True)
        self.pre_context.weight.data.fill_(0)
        self.pre_context.bias.data.fill_(0)
        self.post_context = torch.nn.Conv2d(
            self.out_channels,
            self.out_channels,
            kernel_size=1,
            bias=True)
        self.post_context.weight.data.fill_(0)
        self.post_context.bias.data.fill_(0)

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        # pre-context
        avg_x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=1)
        avg_x = self.pre_context(avg_x)
        avg_x = avg_x.expand_as(x)
        x = x + avg_x
        # switch
        avg_x = torch.nn.functional.pad(x, pad=(2, 2, 2, 2), mode="reflect")
        avg_x = torch.nn.functional.avg_pool2d(avg_x, kernel_size=5, stride=1, padding=0)
        switch = self.switch(avg_x)
        # sac
        weight = self._get_weight(self.weight)
        out_s = super()._conv_forward(x, weight, None)
        ori_p = self.padding
        ori_d = self.dilation
        self.padding = tuple(3 * p for p in self.padding)
        self.dilation = tuple(3 * d for d in self.dilation)
        weight = weight + self.weight_diff
        out_l = super()._conv_forward(x, weight, None)
        out = switch * out_s + (1 - switch) * out_l
        self.padding = ori_p
        self.dilation = ori_d
        # post-context
        avg_x = torch.nn.functional.adaptive_avg_pool2d(out, output_size=1)
        avg_x = self.post_context(avg_x)
        avg_x = avg_x.expand_as(out)
        out = out + avg_x
        return self.act(self.bn(out))


# -----------------------------------------------DSConv----------------------------------------------------
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair

class DSConv(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=None, dilation=1, groups=1, padding_mode='zeros', bias=False, block_size=32, KDSBias=False, CDS=False):
        padding = _pair(autopad(kernel_size, padding))
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        dilation = _pair(dilation)

        blck_numb = math.ceil(((in_channels)/(block_size*groups)))
        super(DSConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        # KDS weight From Paper
        self.intweight = torch.Tensor(out_channels, in_channels, *kernel_size)
        self.alpha = torch.Tensor(out_channels, blck_numb, *kernel_size)

        # KDS bias From Paper
        self.KDSBias = KDSBias
        self.CDS = CDS

        if KDSBias:
            self.KDSb = torch.Tensor(out_channels, blck_numb, *kernel_size)
        if CDS:
            self.CDSw = torch.Tensor(out_channels)
            self.CDSb = torch.Tensor(out_channels)

        self.reset_parameters()

    def get_weight_res(self):
        # Include expansion of alpha and multiplication with weights to include in the convolution layer here
        alpha_res = torch.zeros(self.weight.shape).to(self.alpha.device)

        # Include KDSBias
        if self.KDSBias:
            KDSBias_res = torch.zeros(self.weight.shape).to(self.alpha.device)

        # Handy definitions:
        nmb_blocks = self.alpha.shape[1]
        total_depth = self.weight.shape[1]
        bs = total_depth//nmb_blocks

        llb = total_depth-(nmb_blocks-1)*bs

        # Casting the Alpha values as same tensor shape as weight
        for i in range(nmb_blocks):
            length_blk = llb if i==nmb_blocks-1 else bs

            shp = self.alpha.shape # Notice this is the same shape for the bias as well
            to_repeat=self.alpha[:, i, ...].view(shp[0],1,shp[2],shp[3]).clone()
            repeated = to_repeat.expand(shp[0], length_blk, shp[2], shp[3]).clone()
            alpha_res[:, i*bs:(i*bs+length_blk), ...] = repeated.clone()

            if self.KDSBias:
                to_repeat = self.KDSb[:, i, ...].view(shp[0], 1, shp[2], shp[3]).clone()
                repeated = to_repeat.expand(shp[0], length_blk, shp[2], shp[3]).clone()
                KDSBias_res[:, i*bs:(i*bs+length_blk), ...] = repeated.clone()

        if self.CDS:
            to_repeat = self.CDSw.view(-1, 1, 1, 1)
            repeated = to_repeat.expand_as(self.weight)
            print(repeated.shape)

        # Element-wise multiplication of alpha and weight
        weight_res = torch.mul(alpha_res, self.weight)
        if self.KDSBias:
            weight_res = torch.add(weight_res, KDSBias_res)
        return weight_res

    def forward(self, input):
        # Get resulting weight
        #weight_res = self.get_weight_res()

        # Returning convolution
        return F.conv2d(input, self.weight, self.bias,
                            self.stride, self.padding, self.dilation,
                            self.groups)

class DSConv2D(Conv):
    def __init__(self, inc, ouc, k=1, s=1, p=None, g=1, act=True):
        super().__init__(inc, ouc, k, s, p, g, act)
        self.conv = DSConv(inc, ouc, k, s, p, g)


# ----------------------------------------------------可变形卷积---------------------------------------------------------
class DCNv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=1, groups=1, act=True, dilation=1, deformable_groups=1):
        super(DCNv2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (autopad(kernel_size, padding), autopad(kernel_size, padding))
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        out_channels_offset_mask = (self.deformable_groups * 3 *
                                    self.kernel_size[0] * self.kernel_size[1])
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            out_channels_offset_mask,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.reset_parameters()

    def forward(self, x):
        offset_mask = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        x = torch.ops.torchvision.deform_conv2d(
            x,
            self.weight,
            offset,
            mask,
            self.bias,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            self.groups,
            self.deformable_groups,
            True
        )
        x = self.bn(x)
        x = self.act(x)
        return x

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        std = 1. / math.sqrt(n)
        self.weight.data.uniform_(-std, std)
        self.bias.data.zero_()
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()


#------------------------------------------------------------PConv------------------------------------------------------------------
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




# -------------------------------------------------全维动态卷积------------------------------------------------------------
class od_Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(od_Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)


class ODConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=1, dilation=1, groups=1,
                 reduction=0.0625, kernel_num=4):
        super(ODConv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = od_Attention(in_planes, out_planes, kernel_size, groups=groups,
                                   reduction=reduction, kernel_num=kernel_num)
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes//groups, kernel_size, kernel_size),
                                   requires_grad=True)
        self._initialize_weights()

        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    def _forward_impl_common(self, x):
        # Multiplying channel attention (or filter attention) to weights and feature maps are equivalent,
        # while we observe that when using the latter method the models will run faster with less gpu memory cost.
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x * channel_attention
        x = x.reshape(1, -1, height, width)
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = output * filter_attention
        return output

    def _forward_impl_pw1x(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output

    def forward(self, x):
        return self._forward_impl(x)


# ------------------------------------------------yolov8--------------------------------------------------
class v8_Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(v8_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

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



# ---------------------------MobileNext Begin---------------------------

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class SGBlock(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, keep_3x3=False, initialize_weights=True):
        super(SGBlock, self).__init__()
        assert stride in [1, 2]

        hidden_dim = inp // expand_ratio
        if hidden_dim < oup / 6.:
            hidden_dim = math.ceil(oup / 6.)
            hidden_dim = _make_divisible(hidden_dim, 16)  # + 16

        # self.relu = nn.ReLU6(inplace=True)
        self.identity = False
        self.identity_div = 1
        self.initialize_weights = initialize_weights
        self.expand_ratio = expand_ratio

        self.conv = nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU6(inplace=True),
            # pw
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            # pw
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True),
        )

        if expand_ratio == 2:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(oup, oup, 3, stride, 1, groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )
        elif inp != oup and stride == 1 and keep_3x3 == False:
            self.conv = nn.Sequential(
                # pw-linear
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
            )
        elif inp != oup and stride == 2 and keep_3x3 == False:
            self.conv = nn.Sequential(
                # pw-linear
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(oup, oup, 3, stride, 1, groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )
        elif self.initialize_weights:
            self._initialize_weights()
        else:
            if keep_3x3 == False:
                self.identity = True
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # nn.ReLU6(inplace=True),
                # pw
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(oup, oup, 3, 1, 1, groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv(x)

        if self.identity:
            return out + x
        else:
            return out

# ---------------------------MobileNext End---------------------------




