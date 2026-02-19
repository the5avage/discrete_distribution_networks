# train_ddn.py

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from PIL import Image
import glob


# =========================

# Config and utilities

# =========================

@dataclass
class ExperimentConfig:
    experiment_name: str
    output_dir: str
    seed: int
    device: str

    dataset: Dict[str, Any]
    model: Dict[str, Any]
    training: Dict[str, Any]


def load_config(path: str) -> ExperimentConfig:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg.get("experiment_name", os.path.splitext(os.path.basename(path))[0])
    output_dir = cfg.get("output_dir", os.path.join("experiments", exp_name))
    os.makedirs(output_dir, exist_ok=True)

    seed = cfg.get("seed", 42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    return ExperimentConfig(
        experiment_name=exp_name,
        output_dir=output_dir,
        seed=seed,
        device=device,
        dataset=cfg["dataset"],
        model=cfg["model"],
        training=cfg["training"],
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================

# Datasets

# =========================

class FFHQDataset(Dataset):
    """
    Simple dataset that loads all PNG images from a folder.
    root: folder with *.png
    """
    def __init__(self, root: str, transform=None):
        self.root = root
        self.paths = sorted(glob.glob(os.path.join(root, "*.png")))
        if len(self.paths) == 0:
            raise RuntimeError(f"No PNG images found in {root}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        # no label; return dummy 0
        return img, 0


def build_transforms(dataset_cfg: Dict[str, Any]) -> transforms.Compose:
    image_size = dataset_cfg.get("image_size")
    normalize = dataset_cfg.get("normalize", False)
    mean = dataset_cfg.get("mean", None)
    std = dataset_cfg.get("std", None)
    channels = dataset_cfg.get("channels", None)

    tfms: List[Any] = []
    if image_size is not None:
        tfms.append(transforms.Resize((image_size, image_size)))
    tfms.append(transforms.ToTensor())

    # If MNIST & channels==3, we can optionally convert gray->RGB later if needed,
    # but here we keep raw channels and let model use in_channels.

    if normalize:
        assert mean is not None and std is not None, "normalize=True but mean/std not provided"
        tfms.append(transforms.Normalize(mean, std))

    return transforms.Compose(tfms)


def build_dataloaders(cfg: ExperimentConfig):
    ds_cfg = cfg.dataset
    name = ds_cfg["name"].lower()
    root = ds_cfg.get("root", "./data")
    split_cfg = ds_cfg.get("split", {})
    val_fraction = split_cfg.get("val_fraction", 0.5)
    train_fraction = split_cfg.get("train_fraction", None)
    split_seed = split_cfg.get("seed", cfg.seed)

    tf = build_transforms(ds_cfg)

    if name == "mnist":
        train_dataset = datasets.MNIST(root=root, train=True, transform=tf, download=True)
        full_test_dataset = datasets.MNIST(root=root, train=False, transform=tf, download=True)

        # split test into val / test by val_fraction
        test_size = len(full_test_dataset)
        indices = list(range(test_size))
        rng = np.random.RandomState(split_seed)
        rng.shuffle(indices)

        val_size = int(test_size * val_fraction)
        val_indices = indices[:val_size]
        test_indices = indices[val_size:]

        val_dataset = Subset(full_test_dataset, val_indices)
        test_dataset = Subset(full_test_dataset, test_indices)

    elif name == "cifar10":
        train_dataset = datasets.CIFAR10(root=root, train=True, transform=tf, download=True)
        full_test_dataset = datasets.CIFAR10(root=root, train=False, transform=tf, download=True)

        test_size = len(full_test_dataset)
        indices = list(range(test_size))
        rng = np.random.RandomState(split_seed)
        rng.shuffle(indices)

        val_size = int(test_size * val_fraction)
        val_indices = indices[:val_size]
        test_indices = indices[val_size:]

        val_dataset = Subset(full_test_dataset, val_indices)
        test_dataset = Subset(full_test_dataset, test_indices)

    elif name == "ffhq":
        # All images in a single folder; split into train/val/test from that
        full_dataset = FFHQDataset(root=root, transform=tf)
        N = len(full_dataset)
        indices = list(range(N))
        rng = np.random.RandomState(split_seed)
        rng.shuffle(indices)

        if train_fraction is None:
            # default: 90% train, (1 - train_fraction)*val_fraction for val, rest test
            train_fraction = 0.9
        val_fraction_local = val_fraction
        train_end = int(N * train_fraction)
        val_end = train_end + int(N * val_fraction_local)

        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        test_dataset = Subset(full_dataset, test_indices)

    else:
        raise ValueError(f"Unknown dataset name: {name}")

    tr_cfg = cfg.training
    batch_size = tr_cfg.get("batch_size", 128)
    batch_size_val = tr_cfg.get("batch_size_val", batch_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_val, shuffle=False)

    # channels / size
    channels = ds_cfg.get("channels", None)
    image_size = ds_cfg.get("image_size", None)
    if channels is None:
        # infer from one sample
        sample, _ = train_dataset[0]
        channels = sample.shape[0]
    if image_size is None:
        sample, _ = train_dataset[0]
        image_size = sample.shape[1]  # assume square

    return train_loader, val_loader, test_loader, channels, image_size


# =========================

# Models

# =========================

class MLPImageOnlyDDN(nn.Module):
    """
    Recurrence-iteration DDN with arbitrary K.
    State: x_l only (no persistent feature).
    MLP on flattened image.
    """
    def __init__(self, image_dim: int, hidden_dims=(1024, 512, 256), K: int = 2):
        super().__init__()
        self.image_dim = image_dim
        self.K = K

        self.backbone = nn.Sequential(
            nn.Linear(image_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.LayerNorm(hidden_dims[2]),
            nn.ReLU(),
        )

        self.heads = nn.ModuleList([
            nn.Linear(hidden_dims[-1], image_dim)
            for _ in range(K)
        ])

    def forward(self, x):
        """
        x: (B,C,H,W)
        returns: (B,K,C,H,W)
        """
        B = x.size(0)
        x_flat = x.view(B, -1)
        feat = self.backbone(x_flat)

        outs = []
        for head in self.heads:
            out_flat = head(feat) + x_flat
            outs.append(out_flat.view(B, *x.shape[1:]))

        return torch.stack(outs, dim=1)


class MLPFeatureImageDDN(nn.Module):
    """
    Recurrence-iteration DDN with arbitrary K, carrying a feature vector h_l.
    State: (x_l, h_l).
    """
    def __init__(self,
                 image_dim: int,
                 feat_dim: int = 512,
                 hidden_dims=(1024, 512),
                 K: int = 2):
        super().__init__()
        self.image_dim = image_dim
        self.feat_dim = feat_dim
        self.K = K

        in_dim = image_dim + feat_dim

        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(),
        )

        self.heads = nn.ModuleList([
            nn.Linear(feat_dim, image_dim)
            for _ in range(K)
        ])

    def forward(self, x, h):
        """
        x: (B,C,H,W)
        h: (B,feat_dim)
        returns: (B,K,C,H,W), h_next
        """
        B = x.size(0)
        x_flat = x.view(B, -1)
        inp = torch.cat([x_flat, h], dim=1)
        h_next = self.backbone(inp)

        outs = []
        for head in self.heads:
            out_flat = head(h_next) + x_flat
            outs.append(out_flat.view(B, *x.shape[1:]))

        return torch.stack(outs, dim=1), h_next


class ConvSimpleImageOnlyDDN(nn.Module):
    """
    Simple conv backbone + K 1x1 heads.
    State: x_l only.
    """
    def __init__(self, in_channels: int, base_channels: int = 64, K: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.K = K

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.heads = nn.ModuleList([
            nn.Conv2d(base_channels, in_channels, kernel_size=1)
            for _ in range(K)
        ])

    def forward(self, x):
        feat = self.backbone(x)
        outs = []
        for head in self.heads:
            out = head(feat) + x
            outs.append(out)
        return torch.stack(outs, dim=1)


class ConvSimpleFeatureImageDDN(nn.Module):
    """
    Simple conv backbone with persistent feature map h_l.
    State: (x_l, h_l).
    """
    def __init__(self, in_channels: int, feat_channels: int = 64, K: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.K = K

        in_ch = in_channels + feat_channels
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, feat_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.heads = nn.ModuleList([
            nn.Conv2d(feat_channels, in_channels, kernel_size=1)
            for _ in range(K)
        ])

    def forward(self, x, h):
        inp = torch.cat([x, h], dim=1)
        h_next = self.backbone(inp)
        outs = []
        for head in self.heads:
            out = head(h_next) + x
            outs.append(out)
        return torch.stack(outs, dim=1), h_next


class ConvDDNImageOnly(nn.Module):
    """
    A deeper UNet-like conv backbone closer to the paper's generator.
    Still used in recurrence-iteration fashion (weights shared across steps).
    """
    def __init__(self, in_channels: int, base_channels: int = 64, K: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.K = K

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1)

        self.enc2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        self.dec1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # K heads: 1x1 convs + residual
        self.heads = nn.ModuleList([
            nn.Conv2d(base_channels, in_channels, kernel_size=1)
            for _ in range(K)
        ])

    def forward(self, x):
        # UNet forward
        e1 = self.enc1(x)                 # (B,C,H,W)
        d1 = self.down1(e1)               # (B,2C,H/2,W/2)

        e2 = self.enc2(d1)               # (B,2C,H/2,W/2)
        d2 = self.down2(e2)              # (B,4C,H/4,W/4)

        b = self.bottleneck(d2)          # (B,4C,H/4,W/4)

        u2 = self.up2(b)                 # (B,2C,H/2,W/2)
        u2 = torch.cat([u2, e2], dim=1)  # (B,4C,H/2,W/2)
        d_dec2 = self.dec2(u2)           # (B,2C,H/2,W/2)

        u1 = self.up1(d_dec2)           # (B,C,H,W)
        u1 = torch.cat([u1, e1], dim=1) # (B,2C,H,W)
        d_dec1 = self.dec1(u1)          # (B,C,H,W)

        outs = []
        for head in self.heads:
            out = head(d_dec1) + x
            outs.append(out)
        return torch.stack(outs, dim=1)


class ConvDDNFeatureImage(nn.Module):
    """
    DDN-like conv backbone with persistent feature map h_l.
    """
    def __init__(self, in_channels: int, feat_channels: int = 64, K: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.K = K

        in_ch = in_channels + feat_channels
        base = feat_channels

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.down1 = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)

        self.enc2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1)

        self.bottleneck = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 4, base * 4, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 4, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.dec1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.heads = nn.ModuleList([
            nn.Conv2d(base, in_channels, kernel_size=1)
            for _ in range(K)
        ])

    def forward(self, x, h):
        inp = torch.cat([x, h], dim=1)

        e1 = self.enc1(inp)
        d1 = self.down1(e1)

        e2 = self.enc2(d1)
        d2 = self.down2(e2)

        b = self.bottleneck(d2)

        u2 = self.up2(b)
        u2 = torch.cat([u2, e2], dim=1)
        d_dec2 = self.dec2(u2)

        u1 = self.up1(d_dec2)
        u1 = torch.cat([u1, e1], dim=1)
        d_dec1 = self.dec1(u1)

        h_next = d_dec1  # feature map for next step

        outs = []
        for head in self.heads:
            out = head(h_next) + x
            outs.append(out)
        return torch.stack(outs, dim=1), h_next


# =========================

# Split-and-Prune helper

# =========================

class SplitPruneHelper:
    """
    Simplified Split-and-Prune:
    - Track how often each head k is selected.
    - If some head is overused and another underused, copy params from overused

      to underused (with small noise).
    """

    def __init__(self, K, device, P_split=None, P_prune=None, min_total=5000):
        self.K = K
        self.device = device
        self.counts = torch.zeros(K, device=device)
        self.total = 0.0
        self.P_split = 2.0 / K if P_split is None else P_split
        self.P_prune = 0.5 / K if P_prune is None else P_prune
        self.min_total = min_total

    @torch.no_grad()
    def accumulate(self, best_k):
        batch_counts = torch.bincount(best_k, minlength=self.K).float()
        self.counts += batch_counts
        self.total += batch_counts.sum()

    @torch.no_grad()
    def maybe_split_prune(self, model):
        if self.total < self.min_total:
            return

        probs = self.counts / self.total
        k_max = int(torch.argmax(probs))
        k_min = int(torch.argmin(probs))

        if probs[k_max] > self.P_split or probs[k_min] < self.P_prune:
            src = model.heads[k_max]
            dst = model.heads[k_min]

            dst.weight.data.copy_(src.weight.data)
            if dst.bias is not None and src.bias is not None:
                dst.bias.data.copy_(src.bias.data)

            self.total -= self.counts[k_min].item()
            self.counts[k_min] = 0.0

            noise_scale = 1e-4
            dst.weight.data.add_(noise_scale * torch.randn_like(dst.weight.data))
            if dst.bias is not None:
                dst.bias.data.add_(noise_scale * torch.randn_like(dst.bias.data))


# =========================

# Build model / loss

# =========================

def build_model(cfg: ExperimentConfig, channels: int, image_size: int):
    m_cfg = cfg.model
    mode = m_cfg["mode"]           # "image_only" or "feature_image"
    arch = m_cfg["arch"]           # "mlp", "cnn_simple", "cnn_ddn"
    K = m_cfg["K"]

    if arch == "mlp":
        image_dim = channels * image_size * image_size
        if mode == "image_only":
            hidden_dims = m_cfg.get("mlp", {}).get("hidden_dims", [1024, 512, 256])
            model = MLPImageOnlyDDN(image_dim=image_dim, hidden_dims=hidden_dims, K=K)
        elif mode == "feature_image":
            mlp_cfg = m_cfg.get("mlp", {})
            hidden_dims = mlp_cfg.get("hidden_dims", [1024, 512])
            feat_dim = mlp_cfg.get("feat_dim", 512)
            model = MLPFeatureImageDDN(
                image_dim=image_dim,
                feat_dim=feat_dim,
                hidden_dims=hidden_dims,
                K=K
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    elif arch == "cnn_simple":
        if mode == "image_only":
            base_channels = m_cfg.get("cnn_simple", {}).get("base_channels", 64)
            model = ConvSimpleImageOnlyDDN(in_channels=channels, base_channels=base_channels, K=K)
        elif mode == "feature_image":
            cnn_cfg = m_cfg.get("cnn_simple", {})
            feat_channels = cnn_cfg.get("feat_channels", 64)
            model = ConvSimpleFeatureImageDDN(in_channels=channels, feat_channels=feat_channels, K=K)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    elif arch == "cnn_ddn":
        if mode == "image_only":
            base_channels = m_cfg.get("cnn_ddn", {}).get("base_channels", 64)
            model = ConvDDNImageOnly(in_channels=channels, base_channels=base_channels, K=K)
        elif mode == "feature_image":
            cnn_cfg = m_cfg.get("cnn_ddn", {})
            feat_channels = cnn_cfg.get("feat_channels", 64)
            model = ConvDDNFeatureImage(in_channels=channels, feat_channels=feat_channels, K=K)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    else:
        raise ValueError(f"Unknown arch: {arch}")

    return model


def build_loss(cfg: ExperimentConfig):
    loss_type = cfg.training.get("loss_type", "mse").lower()
    if loss_type == "mse":
        criterion = nn.MSELoss(reduction="none")
    elif loss_type == "bce":
        criterion = nn.BCEWithLogitsLoss(reduction="none")
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    return criterion, loss_type


def build_optimizer_scheduler(cfg: ExperimentConfig, model: nn.Module):
    tr_cfg = cfg.training

    # Raw values from YAML (could be str or numeric)
    lr_raw = tr_cfg.get("learning_rate", 1e-4)
    wd_raw = tr_cfg.get("weight_decay", 0.0)
    warmup_raw = tr_cfg.get("warmup_steps", 0)

    # Robust casting
    try:
        lr = float(lr_raw)
    except Exception:
        raise ValueError(f"Invalid learning_rate value in config: {lr_raw!r}")

    try:
        wd = float(wd_raw)
    except Exception:
        raise ValueError(f"Invalid weight_decay value in config: {wd_raw!r}")

    try:
        warmup_steps = int(warmup_raw)
    except Exception:
        raise ValueError(f"Invalid warmup_steps value in config: {warmup_raw!r}")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0

    scheduler = LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler
# =========================

# Training / Evaluation

# =========================

def init_x0(cfg: ExperimentConfig, shape, device):
    init_type = cfg.training.get("init_type", "zeros")
    if init_type == "zeros":
        return torch.zeros(shape, device=device)
    elif init_type == "noise":
        # standard normal; reasonable for normalized MSE training
        return torch.randn(shape, device=device)
    else:
        raise ValueError(f"Unknown init_type: {init_type}")


def init_h0(cfg: ExperimentConfig, model: nn.Module, B: int, channels: int, image_size: int, device):
    mode = cfg.model["mode"]
    arch = cfg.model["arch"]
    if mode != "feature_image":
        return None

    if arch == "mlp":
        feat_dim = model.feat_dim
        return torch.zeros(B, feat_dim, device=device)
    else:  # cnn_simple or cnn_ddn
        feat_channels = getattr(model, "feat_channels", 64)
        return torch.zeros(B, feat_channels, image_size, image_size, device=device)


def train_epoch(
    cfg: ExperimentConfig,
    model: nn.Module,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    device,
    L: int,
    criterion,
    loss_type: str,
    sp_helper: SplitPruneHelper = None,
):
    model.train()
    total_loss = 0.0

    mode = cfg.model["mode"]
    arch = cfg.model["arch"]

    tr_cfg = cfg.training
    strategy = tr_cfg.get("training_strategy", "naive")

    softmax_tau_raw = tr_cfg.get("softmax_tau", 0.1)
    chain_dropout_raw = tr_cfg.get("chain_dropout_prob", 0.0)
    K_raw = cfg.model["K"]

    try:
        softmax_tau = float(softmax_tau_raw)
    except Exception:
        raise ValueError(f"Invalid softmax_tau in config: {softmax_tau_raw!r}")

    try:
        chain_dropout_prob = float(chain_dropout_raw)
    except Exception:
        raise ValueError(f"Invalid chain_dropout_prob in config: {chain_dropout_raw!r}")

    try:
        K = int(K_raw)
    except Exception:
        raise ValueError(f"Invalid K in config: {K_raw!r}")

    for imgs, _ in dataloader:
        imgs = imgs.to(device)
        B, C, H, W = imgs.shape

        x = init_x0(cfg, imgs.shape, device)
        h = init_h0(cfg, model, B, C, H, device)

        layer_losses = []
        optimizer.zero_grad()

        for l in range(L):
            if mode == "image_only":
                candidates = model(x)  # (B,K,C,H,W)
            else:
                candidates, h = model(x, h)

            targets = imgs.unsqueeze(1).expand_as(candidates)
            loss_per_k = criterion(candidates, targets).mean(dim=(2, 3, 4))  # (B,K)

            if strategy in ("naive", "split_prune"):
                guided_k = loss_per_k.argmin(dim=1)  # (B,)
                if chain_dropout_prob > 0.0:
                    # per-sample chain dropout
                    drop_mask = (torch.rand(B, device=device) < chain_dropout_prob)
                    random_k = torch.randint(0, K, (B,), device=device)
                    best_k = torch.where(drop_mask, random_k, guided_k)
                else:
                    best_k = guided_k

                batch_idx = torch.arange(B, device=device)
                x = candidates[batch_idx, best_k]

                chosen_loss = loss_per_k[batch_idx, best_k].mean()

                if sp_helper is not None and strategy == "split_prune":
                    sp_helper.accumulate(best_k)

            elif strategy == "softmax":
                weights = torch.softmax(-loss_per_k / softmax_tau, dim=1)
                x = (weights.view(B, K, 1, 1, 1) * candidates).sum(dim=1)
                chosen_loss = (weights * loss_per_k).sum(dim=1).mean()
            else:
                raise ValueError(f"Unknown training_strategy: {strategy}")

            layer_losses.append(chosen_loss)

        loss = sum(layer_losses) / len(layer_losses)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if sp_helper is not None and strategy == "split_prune":
            sp_helper.maybe_split_prune(model)

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def eval_epoch(
    cfg: ExperimentConfig,
    model: nn.Module,
    dataloader: DataLoader,
    device,
    L: int,
    criterion,
    loss_type: str,
):
    model.eval()
    total_loss = 0.0

    mode = cfg.model["mode"]
    arch = cfg.model["arch"]
    tr_cfg = cfg.training
    strategy = tr_cfg.get("training_strategy", "naive")

    softmax_tau_raw = tr_cfg.get("softmax_tau", 0.1)
    K_raw = cfg.model["K"]

    try:
        softmax_tau = float(softmax_tau_raw)
    except Exception:
        raise ValueError(f"Invalid softmax_tau in config: {softmax_tau_raw!r}")

    try:
        K = int(K_raw)
    except Exception:
        raise ValueError(f"Invalid K in config: {K_raw!r}")

    for imgs, _ in dataloader:
        imgs = imgs.to(device)
        B, C, H, W = imgs.shape

        x = init_x0(cfg, imgs.shape, device)
        h = init_h0(cfg, model, B, C, H, device)

        layer_losses = []

        for l in range(L):
            if mode == "image_only":
                candidates = model(x)
            else:
                candidates, h = model(x, h)

            targets = imgs.unsqueeze(1).expand_as(candidates)
            loss_per_k = criterion(candidates, targets).mean(dim=(2, 3, 4))

            if strategy in ("naive", "split_prune"):
                best_k = loss_per_k.argmin(dim=1)
                batch_idx = torch.arange(B, device=device)
                x = candidates[batch_idx, best_k]
                chosen_loss = loss_per_k[batch_idx, best_k].mean()
            elif strategy == "softmax":
                weights = torch.softmax(-loss_per_k / softmax_tau, dim=1)
                x = (weights.view(B, K, 1, 1, 1) * candidates).sum(dim=1)
                chosen_loss = (weights * loss_per_k).sum(dim=1).mean()
            else:
                raise ValueError(f"Unknown training_strategy: {strategy}")

            layer_losses.append(chosen_loss)

        loss = sum(layer_losses) / len(layer_losses)
        total_loss += loss.item()

    return total_loss / len(dataloader)


# =========================

# Main

# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, channels, image_size = build_dataloaders(cfg)

    model = build_model(cfg, channels, image_size).to(device)
    criterion, loss_type = build_loss(cfg)
    optimizer, scheduler = build_optimizer_scheduler(cfg, model)

    tr_cfg = cfg.training

    tr_cfg = cfg.training

    # Depth L (can be overridden in training section)
    L_raw = tr_cfg.get("L", cfg.model["L"])
    try:
        L = int(L_raw)
    except Exception:
        raise ValueError(f"Invalid L value in config: {L_raw!r}")

    # Epochs
    epochs_raw = tr_cfg.get("epochs", 50)
    try:
        num_epochs = int(epochs_raw)
    except Exception:
        raise ValueError(f"Invalid epochs value in config: {epochs_raw!r}")

    strategy = tr_cfg.get("training_strategy", "naive")
    K = cfg.model["K"]

    if strategy == "split_prune":
        sp_cfg = tr_cfg.get("split_prune", {})
        sp_helper = SplitPruneHelper(
            K,
            device,
            P_split=sp_cfg.get("P_split", None),
            P_prune=sp_cfg.get("P_prune", None),
            min_total=sp_cfg.get("min_total", 5000),
        )
    else:
        sp_helper = None

    print(f"Experiment: {cfg.experiment_name}")
    print(f"Dataset: {cfg.dataset['name']}, channels={channels}, image_size={image_size}")
    print(f"Model: mode={cfg.model['mode']}, arch={cfg.model['arch']}, K={K}, L={L}")
    print(f"Training: loss_type={loss_type}, strategy={strategy}, epochs={num_epochs}")

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_model_path = os.path.join(
        cfg.output_dir,
        f"best_{cfg.experiment_name}.pth"
    )

    global_step = 0
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(
            cfg,
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            L,
            criterion,
            loss_type,
            sp_helper,
        )
        val_loss = eval_epoch(
            cfg,
            model,
            val_loader,
            device,
            L,
            criterion,
            loss_type,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"[Epoch {epoch}] New best val loss: {val_loss:.6f} -> saved to {best_model_path}")

        print(f"Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    # Test with best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    print(f"Loaded best model from {best_model_path} (best val loss={best_val_loss:.6f})")

    test_loss = eval_epoch(
        cfg,
        model,
        test_loader,
        device,
        L,
        criterion,
        loss_type,
    )
    print(f"Test Loss: {test_loss:.6f}")

    # Save results
    results = {
        "experiment_name": cfg.experiment_name,
        "best_val_loss": float(best_val_loss),
        "test_loss": float(test_loss),
        "train_losses": [float(x) for x in train_losses],
        "val_losses": [float(x) for x in val_losses],
        "config_path": args.config,
    }
    results_path = os.path.join(cfg.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()