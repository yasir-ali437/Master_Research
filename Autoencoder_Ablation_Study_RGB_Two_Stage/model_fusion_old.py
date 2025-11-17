import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# ----------------------------
# Edge Extraction (same as before)
# ----------------------------
class EdgeExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[1., 0., -1.],
                           [2., 0., -2.],
                           [1., 0., -1.]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[1., 2., 1.],
                           [0., 0., 0.],
                           [-1., -2., -1.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_kx", kx)
        self.register_buffer("sobel_ky", ky)

    def forward(self, x_1ch):
        gx = F.conv2d(x_1ch, self.sobel_kx, padding=1)
        gy = F.conv2d(x_1ch, self.sobel_ky, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
        mag = mag / mag.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return mag.clamp(0.0, 1.0)

# ----------------------------
# Edge-Guided Attention (same)
# ----------------------------
class EdgeGuidedAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.proj = nn.Conv2d(1, in_channels, 1)
        self.gamma = nn.Parameter(torch.tensor(0.5))

    def forward(self, feat, edge):
        att = torch.sigmoid(self.proj(edge))
        return feat * (1.0 + self.gamma * att)

# ----------------------------
# Encoder (adds mask channel)
# ----------------------------
class FusionEncoder(nn.Module):
    def __init__(self, input_nc=2, base_ch=32, base_ch2=64):
        """
        input_nc: MRI (1) + PET (1) + Mask (1)
        """
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_nc + 1, base_ch, 3, 1),
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

        self.att1 = EdgeGuidedAttention(base_ch)
        self.att2 = EdgeGuidedAttention(base_ch)

    def forward(self, x_2ch, edge):
        x0 = torch.cat([x_2ch, edge], dim=1)
        G11 = self.att1(self.conv1(x0), edge)
        G21 = self.att2(self.conv2(G11), edge)
        # G11 = self.conv1(x_3ch)
        # G21 = self.conv2(G11)
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
class Fusion_Net(nn.Module):
    def __init__(self, base_ch=32, base_ch2=64):
        super().__init__()
        self.edge_extractor = EdgeExtractor()
        self.encoder = FusionEncoder(input_nc=2, base_ch=base_ch, base_ch2=base_ch2)
        self.decoder = FusionDecoder(base_ch=base_ch, base_ch2=base_ch2)

    def forward(self, mri, pet):
        edge = self.edge_extractor(mri)
        x_2ch = torch.cat([mri, pet], dim=1)
        feats = self.encoder(x_2ch, edge)
        fused, edge_pred = self.decoder(feats)
        return fused, edge_pred
