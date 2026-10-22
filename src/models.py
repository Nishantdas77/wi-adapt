"""
src/models.py
The networks used in Task 2 and Task 3 (helper file, do not run).

CSIEncoder : window (batch, 150 time, 256 subcarriers) -> embedding (batch, 128)
             1D ResNet-style CNN: the 256 subcarriers are the input channels,
             the convolution slides along TIME. Global average over time at the end,
             so it accepts any window length (needed for random crops).

    (B,150,256) -> transpose -> (B,256,150)
    stem   Conv1d 256->128, kernel 5, stride 2      (B,128,75)
    block1 ResBlock 128->128                         (B,128,75)
    block2 ResBlock 128->128, stride 2               (B,128,38)
    block3 ResBlock 128->128, stride 2               (B,128,19)
    mean over time -> Linear 128->128                (B,128)   = embedding

ProjectionHead : embedding (128) -> 64, used ONLY during SimCLR training, then dropped.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1d(nn.Module):
    def __init__(self, c_in, c_out, stride=1, kernel=5):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        if stride == 1 and c_in == c_out:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False),
                                      nn.BatchNorm1d(c_out))

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(y + self.skip(x))


class CSIEncoder(nn.Module):
    def __init__(self, n_sub=256, width=128, embed_dim=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_sub, width, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(width), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(
            ResBlock1d(width, width),
            ResBlock1d(width, width, stride=2),
            ResBlock1d(width, width, stride=2))
        self.fc = nn.Linear(width, embed_dim)

    def forward(self, x):                      # x: (B, T, 256)
        x = self.blocks(self.stem(x.transpose(1, 2)))
        return self.fc(x.mean(dim=2))          # (B, embed_dim)


class ProjectionHead(nn.Module):
    def __init__(self, embed_dim=128, proj_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True), nn.Linear(embed_dim, proj_dim))

    def forward(self, h):
        return self.net(h)


# ------------------------------------------------------------------ helpers
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encoder_arch(t2cfg):
    return {"n_sub": 256, "width": t2cfg["width"], "embed_dim": t2cfg["embed_dim"]}


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def save_encoder(path, encoder, arch, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": arch, "state_dict": encoder.state_dict(), **extra}, path)


def load_encoder(path, device=None):
    device = device or get_device()
    ck = torch.load(path, map_location=device, weights_only=False)
    enc = CSIEncoder(**ck["arch"]).to(device)
    enc.load_state_dict(ck["state_dict"])
    enc.eval()
    return enc, ck


def random_encoder(arch, seed, device=None):
    """Same architecture, untrained weights (fixed seed): the 'random weights' baseline."""
    torch.manual_seed(seed)
    enc = CSIEncoder(**arch).to(device or get_device())
    enc.eval()
    return enc


@torch.no_grad()
def embed_windows(encoder, X, device=None, batch_size=256):
    """X: (N, 150, 256) numpy (float16 or float32) -> (N, embed_dim) float32 numpy.
    Encoder in eval mode, no augmentation, full window."""
    device = device or next(encoder.parameters()).device
    encoder.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(np.asarray(X[i:i + batch_size], dtype=np.float32)).to(device)
        out.append(encoder(xb).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, encoder.fc.out_features), np.float32)