import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# ----------------------------
# Encoder (adds mask channel)
# ----------------------------
class FusionEncoder(nn.Module):
    def __init__(self, input_nc=4, base_ch=32, base_ch2=64):
        """
        input_nc: PET (3) + Mask (1)
        """
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_nc , base_ch, 3, 1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_ch, base_ch, 3, 1),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_ch * 2, base_ch2, 3, 2),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_ch2, base_ch2, 3, 1),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_ch2 * 2, base_ch2, 3, 1),
            nn.ReLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_ch2 * 3, base_ch2, 3, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_4ch):
        G11 = self.conv1(x_4ch)
        G21 = self.conv2(G11)
        G31 = self.conv3(torch.cat([G11, G21], 1))
        G41 = self.conv4(G31)
        G51 = self.conv5(torch.cat([G31, G41], 1))
        G61 = self.conv6(torch.cat([G31, G41, G51], 1))
        return [G11, G21, G31, G41, G51, G61]

# ----------------------------
# Decoder (unchanged)
# ----------------------------
class FusionDecoder(nn.Module):
    def __init__(self, base_ch=32, base_ch2=64, output_nc=3):
        super().__init__()
        self.conv66 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(base_ch2, base_ch2, 3), nn.ReLU(True))
        self.conv55 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(base_ch2 * 2, base_ch2, 3), nn.ReLU(True))
        self.conv44 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(base_ch2 * 2, base_ch2, 3), nn.ReLU(True))
        self.conv33 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(base_ch2 * 2, base_ch2, 3), nn.ReLU(True))
        self.conv22 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(base_ch2 + base_ch, base_ch, 3), nn.ReLU(True))
        self.conv11 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(base_ch * 2, output_nc, 3), nn.Sigmoid())
        self.up = nn.Upsample(scale_factor=2, mode="bicubic", align_corners=False)
        self.edge_head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_ch, base_ch, 3), nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_ch, 1, 3), nn.Sigmoid()
        )

    def forward(self, feats):
        G6_2 = self.conv66(feats[5])
        G5_2 = self.conv55(torch.cat([feats[4], G6_2], 1))
        G4_2 = self.conv44(torch.cat([feats[3], G5_2], 1))
        G3_2 = self.conv33(torch.cat([feats[2], G4_2], 1))
        G2_2 = self.conv22(torch.cat([feats[1], self.up(G3_2)], 1))
        fused = self.conv11(torch.cat([feats[0], G2_2], 1))
        edge_pred = self.edge_head(G2_2)
        return fused, edge_pred

# ----------------------------
# FusionNet (Top-level)
# ----------------------------
class Transport_Net(nn.Module):
    def __init__(self, base_ch=32, base_ch2=64):
        super().__init__()
        self.encoder = FusionEncoder(input_nc=4, base_ch=base_ch, base_ch2=base_ch2)
        self.decoder = FusionDecoder(base_ch=base_ch, base_ch2=base_ch2)

    def forward(self, pet, mask):
        x_4ch = torch.cat([pet, mask], dim=1)
        feats = self.encoder(x_4ch)
        fused, edge_pred = self.decoder(feats)
        return fused, edge_pred
