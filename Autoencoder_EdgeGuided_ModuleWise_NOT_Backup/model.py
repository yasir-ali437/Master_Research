import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

# ----------------------------
# Module: Edge Extraction
# ----------------------------
class EdgeExtractor(nn.Module):
    """
    Module: EdgeExtractor
    Computes Sobel edge magnitude from a single-channel MRI image.
    Keeps sobel kernels as registered buffers so they move with the model/device.
    """
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

    def forward(self, x_1ch: torch.Tensor) -> torch.Tensor:
        """
        x_1ch: [B,1,H,W] in [0,1]
        returns edge magnitude normalized per-sample to [0,1]
        """
        gx = F.conv2d(x_1ch, self.sobel_kx, padding=1)
        gy = F.conv2d(x_1ch, self.sobel_ky, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
        amax = mag.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return (mag / amax).clamp(0.0, 1.0)

# ----------------------------
# Module: Edge-Guided Attention
# ----------------------------
class EdgeGuidedAttention(nn.Module):
    """
    Small module that projects a single-channel edge map to a per-channel gate
    and applies a learnable gamma scaling.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(1, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.tensor(0.5))

    def forward(self, feat: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        """
        feat: [B,C,H,W]
        edge: [B,1,H,W]
        returns gated feat
        """
        att = torch.sigmoid(self.proj(edge))
        return feat * (1.0 + self.gamma * att)

# ----------------------------
# Module: Encoder
# ----------------------------
class FusionEncoder(nn.Module):
    """
    Encapsulates the encoder conv blocks and uses EdgeGuidedAttention on the
    first two blocks (as in the original design).
    """
    def __init__(self, input_nc=2, base_ch=32, base_ch2=64):
        super().__init__()
        self.base_ch = base_ch
        self.base_ch2 = base_ch2

        # first block takes input_nc + 1 (edge) channels
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_nc + 1, base_ch, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_ch * 2, base_ch2, kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_ch2, base_ch2, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_ch2 * 2, base_ch2, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_ch2 * 3, base_ch2, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # attention modules for first two blocks
        self.att1 = EdgeGuidedAttention(base_ch)
        self.att2 = EdgeGuidedAttention(base_ch)

    def forward(self, x_2ch: torch.Tensor, edge: torch.Tensor):
        """
        x_2ch: [B,2,H,W]
        edge: [B,1,H,W]
        returns list of encoder features (G11,G21,G31,G41,G51,G61)
        """
        x0 = torch.cat([x_2ch, edge], dim=1)  # -> 3 channels into conv1

        G11 = self.conv1(x0)
        G11 = self.att1(G11, edge)

        G21 = self.conv2(G11)
        G21 = self.att2(G21, edge)

        G31 = self.conv3(torch.cat([G11, G21], 1))
        G41 = self.conv4(G31)
        G51 = self.conv5(torch.cat([G31, G41], 1))
        G61 = self.conv6(torch.cat([G31, G41, G51], 1))

        return [G11, G21, G31, G41, G51, G61]

# ----------------------------
# Module: Decoder & Heads
# ----------------------------
class FusionDecoder(nn.Module):
    """
    Decoder module which performs the upsampling and final fusion head.
    Also contains the auxiliary edge prediction head.
    """
    def __init__(self, base_ch=32, base_ch2=64, output_nc=1):
        super().__init__()
        self.conv66 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_ch2, base_ch2, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        self.conv55 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_ch2 * 2, base_ch2, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        self.conv44 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_ch2 * 2, base_ch2, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        self.conv33 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_ch2 * 2, base_ch2, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        self.conv22 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_ch2 + base_ch, base_ch, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        self.conv11 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_ch * 2, output_nc, kernel_size=3, stride=1),
            nn.Sigmoid()
        )

        self.up = nn.Upsample(scale_factor=2, mode="bicubic", align_corners=False)

        # auxiliary edge head
        self.edge_head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_ch, 1, kernel_size=3, stride=1),
            nn.Sigmoid()
        )

    def forward(self, feats: List[torch.Tensor]):
        """
        feats: list from encoder [G11, G21, G31, G41, G51, G61]
        returns fused image (B,1,H,W) and edge_pred (B,1,H,W)
        """
        G6_2 = self.conv66(feats[5])
        G5_2 = self.conv55(torch.cat([feats[4], G6_2], 1))
        G4_2 = self.conv44(torch.cat([feats[3], G5_2], 1))
        G3_2 = self.conv33(torch.cat([feats[2], G4_2], 1))
        G2_2 = self.conv22(torch.cat([feats[1], self.up(G3_2)], 1))
        fused = self.conv11(torch.cat([feats[0], G2_2], 1))  # [B,1,H,W] in [0,1]

        edge_pred = self.edge_head(G2_2)
        return fused, edge_pred

# ----------------------------
# Module: FusionNet (Top-level model)
# ----------------------------
class Fusion_net(nn.Module):
    """
    Top-level model composed from modules:
    - EdgeExtractor
    - FusionEncoder
    - FusionDecoder

    Keeps same functional behavior as original: computes Sobel edge from MRI,
    uses edge-guided attention in early encoder blocks, returns fused image
    and auxiliary edge head.
    """
    def __init__(self, input_nc=2, output_nc=1, base_ch=32, base_ch2=64):
        super().__init__()
        self.edge_extractor = EdgeExtractor()
        self.encoder = FusionEncoder(input_nc=input_nc, base_ch=base_ch, base_ch2=base_ch2)
        self.decoder = FusionDecoder(base_ch=base_ch, base_ch2=base_ch2, output_nc=output_nc)

    def forward(self, x_2ch: torch.Tensor):
        """
        Full forward (optional) - returns fused image and edge_pred as (fused, edge_pred).
        If you want encoder features + edge used in other workflows, call encoder/edge_extractor separately.
        """
        mri = x_2ch[:, 0:1, :, :]
        edge = self.edge_extractor(mri)
        feats = self.encoder(x_2ch, edge)
        fused, edge_pred = self.decoder(feats)
        return fused, edge_pred

    # Keep separate encoder/decoder helpers for inference compatibility with prior code
    def encode_with_edge(self, x_2ch: torch.Tensor):
        mri = x_2ch[:, 0:1, :, :]
        edge = self.edge_extractor(mri)
        feats = self.encoder(x_2ch, edge)
        return feats, edge

    def decode_from_feats(self, feats: List[torch.Tensor]):
        return self.decoder(feats)