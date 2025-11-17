# ----------------------------
# Module: IO & Image Utilities
# ----------------------------
to_tensor = transforms.ToTensor()      # [0,1], [C,H,W]
to_pil = transforms.ToPILImage()       # expects [C,H,W] or [H,W]

def load_gray(path: str) -> torch.Tensor:
    """Load single-channel image as float tensor [1,H,W] in [0,1]."""
    img = Image.open(path).convert("L")
    return to_tensor(img)

def save_gray(t: torch.Tensor, path: str):
    """Save [1,H,W] or [H,W] in [0,1] to path as 8-bit PNG."""
    t = t.detach()
    if t.dim() == 4:  # [B,1,H,W] -> take first
        t = t[0]
    if t.dim() == 2:
        t = t.unsqueeze(0)  # -> [1,H,W]
    t = t.to(device="cpu", dtype=torch.float32).clamp(0, 1)
    img = to_pil(t)
    img.save(path)

def rgb_to_ycbcr(image: Image.Image):
    """PIL RGB -> Y, Cb, Cr arrays (not normalized)."""
    rgb_array = np.array(image).astype(np.float32)

    transform_matrix = np.array([[0.299, 0.587, 0.114],
                                 [-0.169, -0.331, 0.5],
                                 [0.5, -0.419, -0.081]], dtype=np.float32)

    ycbcr_array = np.dot(rgb_array, transform_matrix.T)

    y_channel = np.clip(ycbcr_array[:, :, 0], 0, 255)
    cb_channel = ycbcr_array[:, :, 1]
    cr_channel = ycbcr_array[:, :, 2]

    return y_channel, cb_channel, cr_channel

def ycbcr_to_rgb(y, cb, cr):
    """Compose Y, Cb, Cr -> PIL RGB (expects arrays)."""
    ycbcr_array = np.stack((y, cb, cr), axis=-1)

    transform_matrix = np.array([[1, 0, 1.402],
                                 [1, -0.344136, -0.714136],
                                 [1, 1.772, 0]], dtype=np.float32)

    rgb_array = np.dot(ycbcr_array, transform_matrix.T)
    rgb_array = np.clip(rgb_array, 0, 255)
    rgb_array = np.round(rgb_array).astype(np.uint8)
    rgb_image = Image.fromarray(rgb_array, mode='RGB')
    return rgb_image

# ----------------------------
# Module: Padding Utilities
# ----------------------------
def pad_to_even(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int,int,int,int]]:
    """Pad [B,C,H,W] so H,W are even. Returns padded tensor and pad tuple (l,r,t,b)."""
    _, _, H, W = x.shape
    pad_r = (2 - W % 2) % 2
    pad_b = (2 - H % 2) % 2
    pad = (0, pad_r, 0, pad_b)  # (left,right,top,bottom)
    if pad_r or pad_b:
        x = F.pad(x, pad, mode="reflect")
    return x, pad

def unpad(x: torch.Tensor, pad: Tuple[int,int,int,int]) -> torch.Tensor:
    l, r, t, b = 0, pad[1], 0, pad[3]
    if r > 0:
        x = x[:, :, :, :-r]
    if b > 0:
        x = x[:, :, :-b, :]
    return x