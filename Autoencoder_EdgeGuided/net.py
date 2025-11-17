import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoFusion_net(nn.Module):
    """
    Edge-guided variant of your TwoFusion_net:
    - Computes Sobel edge magnitude from MRI (channel 0) internally
    - Concats edge map to the raw inputs for the very first conv
    - Applies light edge-guided attention to the first two encoder blocks
    - Adds an auxiliary 'edge head' that predicts MRI edges from decoder features
    - Uses Sigmoid at the last layer so outputs are in [0,1]
    """
    def __init__(self, input_nc=2, output_nc=1):
        super(TwoFusion_net, self).__init__()

        out_channels_def = 32
        out_channels_def2 = 64

        # ---------- fixed Sobel kernels (buffers so they move with .to(device)) ----------
        kx = torch.tensor([[1., 0., -1.],
                           [2., 0., -2.],
                           [1., 0., -1.]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[1., 2., 1.],
                           [0., 0., 0.],
                           [-1., -2., -1.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_kx", kx)
        self.register_buffer("sobel_ky", ky)

        # ---------- edge-guided attention (1x1 conv to per-channel gate) ----------
        self.edge_proj1 = nn.Conv2d(1, out_channels_def, kernel_size=1)
        self.edge_proj2 = nn.Conv2d(1, out_channels_def, kernel_size=1)
        self.gamma1 = nn.Parameter(torch.tensor(0.5))  # gate strength
        self.gamma2 = nn.Parameter(torch.tensor(0.5))

        # ---------- encoder ----------
        # first block takes 2 inputs + 1 edge map = 3 channels
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_nc + 1, out_channels_def, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels_def, out_channels_def, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels_def * 2, out_channels_def2, kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels_def2, out_channels_def2, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels_def2 * 2, out_channels_def2, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels_def2 * 3, out_channels_def2, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # ---------- decoder ----------
        self.conv66 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels_def2, out_channels_def2, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        self.conv55 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels_def2 * 2, out_channels_def2, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        self.conv44 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels_def2 * 2, out_channels_def2, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        self.conv33 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels_def2 * 2, out_channels_def2, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        self.conv22 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels_def2 + out_channels_def, out_channels_def, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        # Final layer -> Sigmoid to keep [0,1]
        self.conv11 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels_def * 2, output_nc, kernel_size=3, stride=1),
            nn.Sigmoid()
        )

        self.up = nn.Upsample(scale_factor=2, mode="bicubic", align_corners=False)

        # ---------- auxiliary edge head (predict MRI edge map from decoder feature) ----------
        self.edge_head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels_def, out_channels_def, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels_def, 1, kernel_size=3, stride=1),
            nn.Sigmoid()
        )

    # ------------------------ helpers ------------------------
    def _sobel_edges(self, x_1ch: torch.Tensor) -> torch.Tensor:
        """
        x_1ch: [B,1,H,W] in [0,1]
        returns edge magnitude normalized per-sample to [0,1]
        """
        gx = F.conv2d(x_1ch, self.sobel_kx, padding=1)
        gy = F.conv2d(x_1ch, self.sobel_ky, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
        # per-image max normalization (robust)
        amax = mag.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return (mag / amax).clamp(0.0, 1.0)

    # ------------------------ encoder/decoder ------------------------
    def encoder(self, x_2ch):
        """
        x_2ch: [B,2,H,W] where x[:,0] = MRI, x[:,1] = PET
        returns feature list + the edge map for optional external use
        """
        mri = x_2ch[:, 0:1, :, :]
        edge = self._sobel_edges(mri)  # [B,1,H,W]

        x0 = torch.cat([x_2ch, edge], dim=1)  # -> 3 channels into conv1

        G11 = self.conv1(x0)
        # edge-guided attention 1
        att1 = torch.sigmoid(self.edge_proj1(edge))
        G11 = G11 * (1.0 + self.gamma1 * att1)

        G21 = self.conv2(G11)
        # edge-guided attention 2
        att2 = torch.sigmoid(self.edge_proj2(edge))
        G21 = G21 * (1.0 + self.gamma2 * att2)

        G31 = self.conv3(torch.cat([G11, G21], 1))
        G41 = self.conv4(G31)
        G51 = self.conv5(torch.cat([G31, G41], 1))
        G61 = self.conv6(torch.cat([G31, G41, G51], 1))

        return [G11, G21, G31, G41, G51, G61], edge

    def decoder(self, f_en_and_edge):
        """
        f_en_and_edge: tuple(list_of_feats, edge_from_encoder)
        returns fused image and auxiliary edge prediction
        """
        f_en, _edge = f_en_and_edge
        G6_2 = self.conv66(f_en[5])
        G5_2 = self.conv55(torch.cat([f_en[4], G6_2], 1))
        G4_2 = self.conv44(torch.cat([f_en[3], G5_2], 1))
        G3_2 = self.conv33(torch.cat([f_en[2], G4_2], 1))
        G2_2 = self.conv22(torch.cat([f_en[1], self.up(G3_2)], 1))
        fused = self.conv11(torch.cat([f_en[0], G2_2], 1))  # [B,1,H,W] in [0,1]

        # auxiliary edge prediction from G2_2 (same spatial size as fused)
        edge_pred = self.edge_head(G2_2)  # [B,1,H,W]
        return fused, edge_pred
