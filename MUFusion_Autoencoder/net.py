import numpy as np
import torch
import math
import torch.nn as nn
# from scipy.misc import imread, imsave, imresize
import torch.nn.functional as F


# class TwoFusion_net(nn.Module):
#     def __init__(self, input_nc=2, output_nc=1):
#         super(TwoFusion_net, self).__init__()

#         kernel_size = 3
#         stride = 1

#         in_channels = 32;
#         out_channels_def = 32;
#         out_channels_def2 = 64;

#         # encoder
#         self.conv1 = nn.Sequential(
#                 nn.ReflectionPad2d(1),
#                 nn.Conv2d(input_nc, out_channels_def,kernel_size=3,stride=1,padding=0),
#                 nn.ReLU());
#         self.conv2 = nn.Sequential(
#                 nn.ReflectionPad2d(1),
#                 nn.Conv2d(in_channels, out_channels_def,kernel_size=3,stride=1,padding=0),
#                 nn.ReLU());
                    
#         self.conv3 = nn.Sequential(
#                 nn.ReflectionPad2d(1),
#                 nn.Conv2d(in_channels*2, out_channels_def2,kernel_size=3,stride=2,padding=0),
#                 nn.ReLU());
                
#         #64c,64x64
                
#         self.conv4 = nn.Sequential(
#                 nn.ReflectionPad2d(1),
#                 nn.Conv2d(out_channels_def2, out_channels_def2,kernel_size=3,stride=1,padding=0),
#                 nn.ReLU());
                
                
#         self.conv5 = nn.Sequential(
#                 nn.ReflectionPad2d(1),
#                 nn.Conv2d(out_channels_def2*2, out_channels_def2,kernel_size=3,stride=1,padding=0),
#                 nn.ReLU());
#         self.conv6 = nn.Sequential(
#                 nn.ReflectionPad2d(1),
#                 nn.Conv2d(out_channels_def2*3, out_channels_def2,kernel_size=3,stride=1,padding=0),
#                 nn.ReLU());                               

#         # decoder
#         self.conv66 = nn.Sequential(
#                 nn.ReflectionPad2d(1),
#                 nn.Conv2d(out_channels_def2, out_channels_def2, kernel_size=3, stride=1),
#                 nn.ReLU());
#         self.conv55 = nn.Sequential(
#                 nn.ReflectionPad2d(1),
#                 nn.Conv2d(out_channels_def2*2, out_channels_def2, kernel_size=3, stride=1),
#                 nn.ReLU());                
#         self.conv44 = nn.Sequential(
#                 nn.ReflectionPad2d(1),
#                 nn.Conv2d(out_channels_def2*2, out_channels_def2, kernel_size=3, stride=1),
#                 nn.ReLU());
                
#         self.conv33 = nn.Sequential(
#                 nn.ReflectionPad2d(1),
#                 nn.Conv2d(out_channels_def2*2, out_channels_def2, kernel_size=3, stride=1),
#                 nn.ReLU());
#         self.conv22 = nn.Sequential(
#                 nn.ReflectionPad2d(1),
#                 nn.Conv2d(out_channels_def2+out_channels_def, out_channels_def, kernel_size=3, stride=1),
#                 nn.ReLU());
#         self.conv11 = nn.Sequential(
#                 nn.ReflectionPad2d(1),
#                 nn.Conv2d(out_channels_def*2, 1, kernel_size=3, stride=1),
#                 nn.Tanh());
#         self.up = nn.Upsample(scale_factor =2,mode="bicubic");
  
#     def encoder(self, input):
#         G11 = self.conv1(input)
#         G21 = self.conv2(G11);
#         G31 = self.conv3(torch.cat([G11,G21],1));
        
#         G41 = self.conv4(torch.cat([G31],1));
#         G51 = self.conv5(torch.cat([G31,G41],1));
#         G61 = self.conv6(torch.cat([G31,G41,G51],1));

#         return [G11,G21,G31,G41,G51,G61]

#     def decoder(self, f_en):
#         G6_2 = self.conv66(torch.cat([f_en[5]],1));
#         G5_2 = self.conv55(torch.cat([f_en[4],G6_2],1));
#         G4_2 = self.conv44(torch.cat([f_en[3],G5_2],1));
        
#         G3_2 = self.conv33(torch.cat([f_en[2],G4_2],1));
#         G2_2 = self.conv22(torch.cat([f_en[1],self.up(G3_2)],1));
#         G1_2 = self.conv11(torch.cat([f_en[0],G2_2],1))
#         return [G1_2]

# ---------- Building blocks ----------

class ConvBlock(nn.Module):
    """Conv2d -> (InstanceNorm) -> SiLU with same spatial size."""
    def __init__(self, in_ch, out_ch, k=3, s=1, norm=True):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=not norm)
        self.norm = nn.InstanceNorm2d(out_ch, affine=True) if norm else nn.Identity()
        self.act  = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class SEBlock(nn.Module):
    """Squeeze-and-Excitation (channel attention)."""
    def __init__(self, ch, r=8):
        super().__init__()
        m = max(ch // r, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, m, 1, bias=True), nn.SiLU(inplace=True),
            nn.Conv2d(m, ch, 1, bias=True), nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(x)
        return x * w

class UpBlock(nn.Module):
    """Bilinear upsample -> ConvBlock (safer than transposed conv default)."""
    def __init__(self, in_ch, out_ch, norm=True):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, k=3, s=1, norm=norm)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.conv(x)

def he_init(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # Older torch doesn't accept 'silu' in kaiming gain; fall back to 'relu'
            try:
                nn.init.kaiming_normal_(m.weight, nonlinearity='silu')
            except ValueError:
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ---------- Improved MUFusion backbone ----------

class TwoFusion_net(nn.Module):
    """
    Backward-compatible shape with your TwoFusion_net, but:
    - InstanceNorm+SiLU in all convs
    - Learnable modality gates before encoder
    - SE attention on deeper features
    - Slightly richer decoder (up+conv at both stages)
    - Single downsample level preserved for compatibility
    """
    def __init__(self, input_nc=2, output_nc=1, base=32):
        super().__init__()
        c1, c2 = base, base         # 32, 32
        c3     = base * 2           # 64

        # --- Modality-gated fusion (learnable weights per channel) ---
        # Assumes 2 modalities stacked in channel dim: [B,2,H,W]
        assert input_nc >= 2, "Expecting 2-channel (two-modality) input."
        self.gate = nn.Sequential(
            nn.Conv2d(input_nc, input_nc, 1, bias=True),
            nn.Sigmoid()
        )

        # --- Encoder (one downsample level like your original) ---
        self.enc1 = ConvBlock(input_nc, c1)            # HxW
        self.enc2 = ConvBlock(c1, c2)                  # HxW
        # anti-aliased-ish downsample: avgpool then stride-1 conv
        self.pre_ds = nn.AvgPool2d(2)                  # H/2 x W/2
        self.enc3 = ConvBlock(c1 + c2, c3)             # H/2 x W/2
        self.enc4 = nn.Sequential(ConvBlock(c3, c3), SEBlock(c3))     # H/2
        self.enc5 = nn.Sequential(ConvBlock(c3 + c3, c3), SEBlock(c3))# H/2
        self.enc6 = nn.Sequential(ConvBlock(c3 + c3 + c3, c3),
                                  SEBlock(c3))         # bottleneck (H/2)

        # --- Decoder (progressive refinement) ---
        self.dec6 = ConvBlock(c3, c3)                                  # H/2
        self.dec5 = ConvBlock(c3 + c3, c3)                             # H/2
        self.dec4 = ConvBlock(c3 + c3, c3)                             # H/2
        self.up3  = UpBlock(c3, c3)                                    # -> H
        # self.dec3 = ConvBlock((c1 + c2) + c3, c3)                      # H
        self.dec3 = ConvBlock(c3 + c2 + c1 + c3, c3)
        self.dec2 = ConvBlock(c2 + c3, c1)                             # H
        self.dec1 = nn.Sequential(
            ConvBlock(c1 + c1, c1),
            nn.Conv2d(c1, output_nc, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # keep Tanh if your targets are scaled to [-1,1]
        )

        he_init(self)

    def encoder(self, x):
        # modality gates (elementwise per-channel)
        g = self.gate(x)            # [B,2,H,W] in simplest case
        x = x * g                   # soft-gated fusion instead of raw concat

        g11 = self.enc1(x)          # H
        g21 = self.enc2(g11)        # H
        h2  = self.pre_ds(torch.cat([g11, g21], dim=1))  # H/2 (concat then blur/down)
        g31 = self.enc3(h2)         # H/2
        g41 = self.enc4(g31)        # H/2
        g51 = self.enc5(torch.cat([g31, g41], dim=1))    # H/2
        g61 = self.enc6(torch.cat([g31, g41, g51], dim=1))  # H/2
        return [g11, g21, g31, g41, g51, g61]

    def decoder(self, f):
        # f: [g11,g21,g31,g41,g51,g61]
        g6_2 = self.dec6(f[5])                            # H/2
        g5_2 = self.dec5(torch.cat([f[4], g6_2], dim=1))  # H/2
        g4_2 = self.dec4(torch.cat([f[3], g5_2], dim=1))  # H/2

        up3  = self.up3(g4_2)                             # -> H
        # rejoin with early encoder features (richer than only f[2])
        g3_2 = self.dec3(torch.cat([F.interpolate(f[2], scale_factor=2, mode='bilinear', align_corners=False),
                                    f[1], f[0], up3], dim=1))  # H
        g2_2 = self.dec2(torch.cat([f[1], g3_2], dim=1))       # H
        g1_2 = self.dec1(torch.cat([f[0], g2_2], dim=1))       # H
        return [g1_2]

    def forward(self, x):
        f = self.encoder(x)
        y = self.decoder(f)[0]
        return y

