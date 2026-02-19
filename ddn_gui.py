# ddn_gui.py

import argparse
import math
import os
import random
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
import tkinter as tk
import glob

# Reuse the same model definitions as in train_ddn.py

# (copy-pasted for a self-contained script)

# =========================

# Model definitions (same as in train_ddn.py)

# =========================

class MLPImageOnlyDDN(nn.Module):
    def __init__(self, image_dim=3*32*32, hidden_dims=(1024, 512, 256), K=2):
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
        B = x.size(0)
        x_flat = x.view(B, -1)
        feat = self.backbone(x_flat)
        outs = []
        for head in self.heads:
            out_flat = head(feat) + x_flat
            outs.append(out_flat.view(B, *x.shape[1:]))
        return torch.stack(outs, dim=1)


class MLPFeatureImageDDN(nn.Module):
    def __init__(self, image_dim=3*32*32, feat_dim=512, hidden_dims=(1024, 512), K=2):
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
    def __init__(self, in_channels=3, base_channels=64, K=2):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.K = K
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
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
            outs.append(head(feat) + x)
        return torch.stack(outs, dim=1)


class ConvSimpleFeatureImageDDN(nn.Module):
    def __init__(self, in_channels=3, feat_channels=64, K=2):
        super().__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.K = K
        in_ch = in_channels + feat_channels
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, feat_channels, 3, padding=1),
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
            outs.append(head(h_next) + x)
        return torch.stack(outs, dim=1), h_next


class ConvDDNImageOnly(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, K=2):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.K = K
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
        self.bottleneck = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.ReLU(inplace=True),
        )
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
        self.heads = nn.ModuleList([
            nn.Conv2d(base_channels, in_channels, kernel_size=1)
            for _ in range(K)
        ])

    def forward(self, x):
        e1 = self.enc1(x)
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
        outs = []
        for head in self.heads:
            outs.append(head(d_dec1) + x)
        return torch.stack(outs, dim=1)


class ConvDDNFeatureImage(nn.Module):
    def __init__(self, in_channels=3, feat_channels=64, K=2):
        super().__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.K = K
        in_ch = in_channels + feat_channels
        base = feat_channels
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
        h_next = d_dec1
        outs = []
        for head in self.heads:
            outs.append(head(h_next) + x)
        return torch.stack(outs, dim=1), h_next


def build_model(model_cfg: Dict[str, Any], channels: int, image_size: int):
    mode = model_cfg["mode"]         # "image_only" or "feature_image"
    arch = model_cfg["arch"]         # "mlp", "cnn_simple", "cnn_ddn"
    K = model_cfg["K"]

    if arch == "mlp":
        image_dim = channels * image_size * image_size
        if mode == "image_only":
            hidden_dims = model_cfg.get("mlp", {}).get("hidden_dims", [1024, 512, 256])
            return MLPImageOnlyDDN(image_dim=image_dim, hidden_dims=hidden_dims, K=K)
        else:
            mlp_cfg = model_cfg.get("mlp", {})
            hidden_dims = mlp_cfg.get("hidden_dims", [1024, 512])
            feat_dim = mlp_cfg.get("feat_dim", 512)
            return MLPFeatureImageDDN(image_dim=image_dim, feat_dim=feat_dim, hidden_dims=hidden_dims, K=K)
    elif arch == "cnn_simple":
        if mode == "image_only":
            base_channels = model_cfg.get("cnn_simple", {}).get("base_channels", 64)
            return ConvSimpleImageOnlyDDN(in_channels=channels, base_channels=base_channels, K=K)
        else:
            cnn_cfg = model_cfg.get("cnn_simple", {})
            feat_channels = cnn_cfg.get("feat_channels", 64)
            return ConvSimpleFeatureImageDDN(in_channels=channels, feat_channels=feat_channels, K=K)
    elif arch == "cnn_ddn":
        if mode == "image_only":
            base_channels = model_cfg.get("cnn_ddn", {}).get("base_channels", 64)
            return ConvDDNImageOnly(in_channels=channels, base_channels=base_channels, K=K)
        else:
            cnn_cfg = model_cfg.get("cnn_ddn", {})
            feat_channels = cnn_cfg.get("feat_channels", 64)
            return ConvDDNFeatureImage(in_channels=channels, feat_channels=feat_channels, K=K)
    else:
        raise ValueError(f"Unknown arch: {arch}")


# =========================

# Dataset / transforms (same semantics as training)

# =========================

class FFHQDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.paths = sorted(glob.glob(os.path.join(root, "*.png")))
        if len(self.paths) == 0:
            raise RuntimeError(f"No PNG images found in {root}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, 0


def build_transforms(dataset_cfg: Dict[str, Any]):
    image_size = dataset_cfg.get("image_size")
    normalize = dataset_cfg.get("normalize", False)
    mean = dataset_cfg.get("mean", None)
    std = dataset_cfg.get("std", None)

    tfms: List[Any] = []
    if image_size is not None:
        tfms.append(transforms.Resize((image_size, image_size)))
    tfms.append(transforms.ToTensor())
    if normalize:
        assert mean is not None and std is not None
        tfms.append(transforms.Normalize(mean, std))
    return transforms.Compose(tfms)


def build_test_dataset(dataset_cfg: Dict[str, Any], seed: int):
    name = dataset_cfg["name"].lower()
    root = dataset_cfg.get("root", "./data")
    split_cfg = dataset_cfg.get("split", {})
    val_fraction = split_cfg.get("val_fraction", 0.5)
    train_fraction = split_cfg.get("train_fraction", None)
    split_seed = split_cfg.get("seed", seed)

    tf = build_transforms(dataset_cfg)

    if name == "mnist":
        full_test_dataset = datasets.MNIST(root=root, train=False, transform=tf, download=True)
        test_size = len(full_test_dataset)
        indices = list(range(test_size))
        rng = np.random.RandomState(split_seed)
        rng.shuffle(indices)
        val_size = int(test_size * val_fraction)
        test_indices = indices[val_size:]
        return Subset(full_test_dataset, test_indices)

    elif name == "cifar10":
        full_test_dataset = datasets.CIFAR10(root=root, train=False, transform=tf, download=True)
        test_size = len(full_test_dataset)
        indices = list(range(test_size))
        rng = np.random.RandomState(split_seed)
        rng.shuffle(indices)
        val_size = int(test_size * val_fraction)
        test_indices = indices[val_size:]
        return Subset(full_test_dataset, test_indices)

    elif name == "ffhq":
        full_dataset = FFHQDataset(root=root, transform=tf)
        N = len(full_dataset)
        indices = list(range(N))
        rng = np.random.RandomState(split_seed)
        rng.shuffle(indices)
        if train_fraction is None:
            train_fraction = 0.9
        train_end = int(N * train_fraction)
        val_end = train_end + int(N * val_fraction)
        test_indices = indices[val_end:]
        return Subset(full_dataset, test_indices)

    else:
        raise ValueError(f"Unknown dataset name: {name}")


# =========================

# GUI

# =========================

class DDN_GUI:
    def __init__(self, master, config, model, device):
        self.master = master
        self.cfg = config
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        self.ds_cfg = config["dataset"]
        self.model_cfg = config["model"]
        self.tr_cfg = config["training"]

        self.loss_type = self.tr_cfg.get("loss_type", "mse").lower()
        if self.loss_type == "mse":
            self.criterion = nn.MSELoss(reduction="none")
        else:
            self.criterion = nn.BCEWithLogitsLoss(reduction="none")

        self.mode_str = self.model_cfg["mode"]
        self.arch_str = self.model_cfg["arch"]
        self.K = self.model_cfg["K"]
        self.L = self.model_cfg["L"]

        self.normalize = self.ds_cfg.get("normalize", False)
        # Store mean/std as 1D tensors; reshape as needed depending on tensor shape
        self.mean = torch.tensor(self.ds_cfg.get("mean", [0.0]), dtype=torch.float32)
        self.std = torch.tensor(self.ds_cfg.get("std", [1.0]), dtype=torch.float32)

        self.init_type = self.tr_cfg.get("init_type", "zeros")

        # Data
        self.test_dataset = build_test_dataset(self.ds_cfg, config.get("seed", 42))
        self.channels = self.ds_cfg.get("channels", None)
        self.image_size = self.ds_cfg.get("image_size", None)
        if self.channels is None or self.image_size is None:
            x0, _ = self.test_dataset[0]
            C, H, W = x0.shape
            if self.channels is None:
                self.channels = C
            if self.image_size is None:
                self.image_size = H

        # GUI-controlled state
        self.current_step = 0
        self.path = []
        self.target = None        # (1,C,H,W)
        self.x = None             # (1,C,H,W)
        self.h = None             # feature
        self.candidates = None    # (1,K,C,H,W)
        self.candidate_images = None
        self.current_losses = None
        self.auto_running = False
        self.auto_after_id = None

        self.gui_mode_var = tk.StringVar(value="reconstruct")
        self.guided_strategy_var = tk.StringVar(value="argmin")

        self._build_layout()
        self._select_new_target()

    def _init_x0(self):
        shape = (1, self.channels, self.image_size, self.image_size)
        if self.init_type == "zeros":
            return torch.zeros(shape, device=self.device)
        elif self.init_type == "noise":
            return torch.randn(shape, device=self.device)
        else:
            raise ValueError(f"Unknown init_type: {self.init_type}")

    def _init_h0(self, batch_size=1):
        if self.mode_str != "feature_image":
            return None
        if self.arch_str == "mlp":
            feat_dim = self.model.feat_dim
            return torch.zeros(batch_size, feat_dim, device=self.device)
        else:
            feat_channels = getattr(self.model, "feat_channels", 64)
            return torch.zeros(batch_size, feat_channels, self.image_size, self.image_size, device=self.device)

    def _reset_state(self):
        self.stop_auto()
        self.current_step = 0
        self.path = []
        self.x = self._init_x0()
        self.h = self._init_h0(batch_size=1)
        self.candidates = None
        self.candidate_images = None
        self.current_losses = None

    # --- layout

    def _build_layout(self):
        ctrl_frame = tk.Frame(self.master)
        ctrl_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Label(ctrl_frame, text="GUI Mode:").grid(row=0, column=0, padx=4)
        tk.Radiobutton(
            ctrl_frame, text="Reconstruction",
            variable=self.gui_mode_var, value="reconstruct",
            command=self._on_gui_mode_change
        ).grid(row=0, column=1, padx=4)
        tk.Radiobutton(
            ctrl_frame, text="Manual Tree",
            variable=self.gui_mode_var, value="manual",
            command=self._on_gui_mode_change
        ).grid(row=0, column=2, padx=4)

        tk.Label(ctrl_frame, text="Guided:").grid(row=0, column=3, padx=4)
        tk.OptionMenu(
            ctrl_frame, self.guided_strategy_var, "argmin", "softmax"
        ).grid(row=0, column=4, padx=4)

        tk.Button(ctrl_frame, text="New Target", command=self._on_new_target).grid(row=1, column=0, padx=4, pady=4)
        tk.Button(ctrl_frame, text="Reset x/h", command=self._on_reset).grid(row=1, column=1, padx=4, pady=4)
        tk.Button(ctrl_frame, text="Step (guided/random)", command=self._on_step_once).grid(row=1, column=2, padx=4, pady=4)
        tk.Button(ctrl_frame, text="Auto Run", command=self._on_auto_run).grid(row=1, column=3, padx=4, pady=4)
        tk.Button(ctrl_frame, text="Stop Auto", command=self.stop_auto).grid(row=1, column=4, padx=4, pady=4)

        self.path_label = tk.Label(self.master, text="Path: (start)")
        self.path_label.pack(pady=2)

        self.info_label = tk.Label(self.master, text="Step: 0")
        self.info_label.pack(pady=2)

        mid_frame = tk.Frame(self.master)
        mid_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Target image figure
        self.fig_target, self.ax_target = plt.subplots(figsize=(2.5, 2.5))
        self.ax_target.axis("off")
        if self.channels == 1:
            init_target = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            self.im_target = self.ax_target.imshow(init_target, vmin=0, vmax=1, cmap="gray")
        else:
            init_target = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
            self.im_target = self.ax_target.imshow(init_target, vmin=0, vmax=1)
        self.canvas_target = FigureCanvasTkAgg(self.fig_target, master=mid_frame)
        self.canvas_target.get_tk_widget().pack(side=tk.LEFT, padx=10)

        # Current x image figure
        self.fig_x, self.ax_x = plt.subplots(figsize=(2.5, 2.5))
        self.ax_x.axis("off")
        if self.channels == 1:
            init_x = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            self.im_x = self.ax_x.imshow(init_x, vmin=0, vmax=1, cmap="gray")
        else:
            init_x = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
            self.im_x = self.ax_x.imshow(init_x, vmin=0, vmax=1)
        self.canvas_x = FigureCanvasTkAgg(self.fig_x, master=mid_frame)
        self.canvas_x.get_tk_widget().pack(side=tk.LEFT, padx=10)

        # Candidate images
        bottom_frame = tk.Frame(self.master)
        bottom_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        n_cols = min(self.K, 8)
        n_rows = int(math.ceil(self.K / n_cols))
        self.fig_cand, self.ax_cand = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[self.ax_cand]])
        elif n_rows == 1:
            axes = self.ax_cand[np.newaxis, :]
        elif n_cols == 1:
            axes = self.ax_cand[:, np.newaxis]
        else:
            axes = self.ax_cand
        self.cand_axes = axes.reshape(-1).tolist()

        self.im_candidates = []
        for i, ax in enumerate(self.cand_axes):
            if i < self.K:
                ax.axis("off")
                if self.channels == 1:
                    init_cand = np.zeros((self.image_size, self.image_size), dtype=np.float32)
                    im = ax.imshow(init_cand, vmin=0, vmax=1, cmap="gray")
                else:
                    init_cand = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
                    im = ax.imshow(init_cand, vmin=0, vmax=1)
                ax.set_title(f"k={i}", fontsize=8)
                self.im_candidates.append(im)
            else:
                ax.axis("off")

        self.canvas_cand = FigureCanvasTkAgg(self.fig_cand, master=bottom_frame)
        self.canvas_cand.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.fig_cand.canvas.mpl_connect("button_press_event", self._on_click_candidate)

    # --- callbacks

    def _on_gui_mode_change(self):
        self._reset_state()
        if self.target is not None:
            self._forward_one_step()
            if self.gui_mode_var.get() == "reconstruct":
                self._compute_losses()
        self._update_all()

    def _on_new_target(self):
        self._select_new_target()

    def _on_reset(self):
        self._reset_state()
        self._forward_one_step()
        if self.gui_mode_var.get() == "reconstruct":
            self._compute_losses()
        self._update_all()

    def _on_step_once(self):
        if self.gui_mode_var.get() == "reconstruct":
            self._guided_step()
        else:
            self._random_step()
        self._update_all()

    def _on_auto_run(self):
        if self.auto_running:
            return
        self.auto_running = True
        self._auto_step_loop()

    def stop_auto(self):
        if self.auto_after_id is not None:
            try:
                self.master.after_cancel(self.auto_after_id)
            except Exception:
                pass
            self.auto_after_id = None
        self.auto_running = False

    def _on_click_candidate(self, event):
        if event.inaxes is None or self.candidates is None:
            return
        for idx, ax in enumerate(self.cand_axes):
            if idx >= self.K:
                break
            if event.inaxes == ax:
                self._choose_branch(idx)
                self._update_all()
                break

    # --- core logic

    def _select_new_target(self):
        idx = random.randrange(len(self.test_dataset))
        sample, _ = self.test_dataset[idx]
        self.target = sample.unsqueeze(0).to(self.device)
        self._reset_state()
        self._forward_one_step()
        if self.gui_mode_var.get() == "reconstruct":
            self._compute_losses()
        self._update_all()

    def _forward_one_step(self):
        self.model.eval()
        with torch.no_grad():
            if self.mode_str == "image_only":
                candidates = self.model(self.x)
            else:
                candidates, h_next = self.model(self.x, self.h)
                self.h = h_next
            self.candidates = candidates  # (1,K,C,H,W)

        B, K_, C, H, W = self.candidates.shape
        assert B == 1 and K_ == self.K

        images = []
        if self.loss_type == "bce" and not self.normalize:
            # BCE logits; visualize via sigmoid to [0,1]
            c = torch.sigmoid(self.candidates.detach().cpu())
        else:
            c = self.candidates.detach().cpu()

        for k in range(self.K):
            img = c[0, k]  # (C,H,W)
            if self.normalize:
                img = img * self.std.view(-1, 1, 1) + self.mean.view(-1, 1, 1)
            img = torch.clamp(img, 0.0, 1.0)
            if self.channels == 1:
                img_np = img[0].numpy()
            else:
                img_np = img.permute(1, 2, 0).numpy()
            images.append(img_np)
        self.candidate_images = images

    def _compute_losses(self):
        if self.target is None or self.candidates is None:
            self.current_losses = None
            return
        with torch.no_grad():
            targets = self.target.unsqueeze(1).expand_as(self.candidates)
            loss_per_k = self.criterion(self.candidates, targets).mean(dim=(2, 3, 4))
            self.current_losses = loss_per_k[0].detach().cpu().numpy()

    def _choose_branch(self, k_idx):
        if self.candidates is None:
            self._forward_one_step()
        with torch.no_grad():
            chosen = self.candidates[0, k_idx:k_idx+1]
            self.x = chosen.to(self.device)
        self.path.append(k_idx)
        self.current_step += 1
        if self.current_step < self.L:
            self._forward_one_step()
            if self.gui_mode_var.get() == "reconstruct":
                self._compute_losses()
        else:
            self.candidates = None
            self.candidate_images = None
            self.current_losses = None

    def _guided_step(self):
        if self.target is None:
            self._select_new_target()
        if self.current_step >= self.L:
            return
        if self.candidates is None:
            self._forward_one_step()
        self._compute_losses()
        if self.current_losses is None:
            return

        strategy = self.guided_strategy_var.get()
        losses = self.current_losses
        if strategy == "argmin":
            k_idx = int(np.argmin(losses))
        else:
            logits = -losses / max(0.1, 1e-8)
            logits = logits - logits.max()
            probs = np.exp(logits)
            probs = probs / probs.sum()
            k_idx = int(np.random.choice(self.K, p=probs))
        self._choose_branch(k_idx)

    def _random_step(self):
        if self.current_step >= self.L:
            return
        if self.candidates is None:
            self._forward_one_step()
        k_idx = random.randrange(self.K)
        self._choose_branch(k_idx)

    def _auto_step_loop(self):
        if not self.auto_running:
            return
        if self.current_step >= self.L:
            self.auto_running = False
            return

        if self.gui_mode_var.get() == "reconstruct":
            self._guided_step()
        else:
            self._random_step()

        self._update_all()
        if self.auto_running and self.current_step < self.L:
            self.auto_after_id = self.master.after(800, self._auto_step_loop)
        else:
            self.auto_running = False
            self.auto_after_id = None

    # --- draw

    def _update_all(self):
        # target
        if self.target is not None:
            t = self.target.detach().cpu()  # (1,C,H,W)
            if self.normalize:
                t = t * self.std.view(1, -1, 1, 1) + self.mean.view(1, -1, 1, 1)
            t = torch.clamp(t, 0.0, 1.0)
            if self.channels == 1:
                img = t[0, 0].numpy()
            else:
                img = t[0].permute(1, 2, 0).numpy()
        else:
            if self.channels == 1:
                img = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            else:
                img = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
        self.im_target.set_data(img)
        self.canvas_target.draw()

        # x
        if self.x is not None:
            x_cpu = self.x.detach().cpu()
            if self.loss_type == "bce" and not self.normalize:
                x_cpu = torch.sigmoid(x_cpu)
            if self.normalize:
                x_cpu = x_cpu * self.std.view(1, -1, 1, 1) + self.mean.view(1, -1, 1, 1)
            x_cpu = torch.clamp(x_cpu, 0.0, 1.0)
            if self.channels == 1:
                imgx = x_cpu[0, 0].numpy()
            else:
                imgx = x_cpu[0].permute(1, 2, 0).numpy()
        else:
            if self.channels == 1:
                imgx = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            else:
                imgx = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
        self.im_x.set_data(imgx)
        self.canvas_x.draw()

        # candidates
        if self.candidate_images is not None:
            for k in range(self.K):
                self.im_candidates[k].set_data(self.candidate_images[k])
                if self.current_losses is not None:
                    loss_val = self.current_losses[k]
                    self.cand_axes[k].set_title(f"k={k}, loss={loss_val:.3f}", fontsize=8)
                else:
                    self.cand_axes[k].set_title(f"k={k}", fontsize=8)
        else:
            for k in range(self.K):
                if self.channels == 1:
                    blank = np.zeros((self.image_size, self.image_size), dtype=np.float32)
                else:
                    blank = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
                self.im_candidates[k].set_data(blank)
                self.cand_axes[k].set_title(f"k={k}", fontsize=8)
        self.canvas_cand.draw()

        if len(self.path) == 0:
            self.path_label.config(text="Path: (start)")
        else:
            self.path_label.config(text="Path: " + "-".join(str(p) for p in self.path))

        mode_text = self.gui_mode_var.get()
        info = f"Step: {self.current_step}/{self.L} | GUI mode: {mode_text}"
        if self.current_losses is not None and len(self.current_losses) > 0:
            best_loss = float(np.min(self.current_losses))
            info += f" | best loss: {best_loss:.4f}"
        self.info_label.config(text=info)


# =========================

# Main

# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights (optional).")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ds_cfg = cfg["dataset"]
    model_cfg = cfg["model"]

    # Build a small dataset to infer shapes
    tmp_ds = build_test_dataset(ds_cfg, seed)
    x0, _ = tmp_ds[0]
    C, H, W = x0.shape
    channels = C
    image_size = H

    model = build_model(model_cfg, channels, image_size)
    weights_path = args.weights
    if weights_path is None:
        # default: use best checkpoint path if running from same output_dir scheme
        exp_name = cfg.get("experiment_name", "exp")
        output_dir = cfg.get("output_dir", os.path.join("experiments", exp_name))
        weights_path = os.path.join(output_dir, f"best_{exp_name}.pth")
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded weights from {weights_path}")

    root = tk.Tk()
    root.title(f"DDN GUI - {cfg.get('experiment_name', '')}")
    app = DDN_GUI(root, cfg, model, device)
    root.mainloop()


if __name__ == "__main__":
    main()