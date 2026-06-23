"""
model.py — CNN1d for drone type classification from spectral scalars

Input  : (batch, 1, 14) float32  — 7 H-band + 7 L-band spectral scalars
                                    treated as a 1-channel sequence of length 14
Output : (batch, 4)     float32  — logits for [background, bebop, ar, phantom]

Kept in a standalone file so:
  - train.py imports it
  - eval_onnx.py does NOT need it (ONNX is self-contained)
  - drone-edge TensorRT wrapper never touches this file
"""

import torch
import torch.nn as nn


class DroneCNN1dClassifier(nn.Module):
    def __init__(
        self,
        in_channels:  int   = 1,
        seq_len:      int   = 14,
        num_filters:  int   = 64,
        kernel_size:  int   = 3,
        num_classes:  int   = 4,
        dropout:      float = 0.3,
    ):
        super().__init__()

        # ── Convolutional backbone ────────────────────────────────────────
        # Block 1: (B, 1, 14) → (B, num_filters, 14)
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels, num_filters, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Block 2: (B, num_filters, 14) → (B, num_filters*2, 14)
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(num_filters, num_filters * 2, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(num_filters * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Block 3: (B, num_filters*2, 14) → (B, num_filters*2, 14)
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(num_filters * 2, num_filters * 2, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(num_filters * 2),
            nn.ReLU(),
        )

        # ── Global Average Pooling → (B, num_filters*2) ──────────────────
        self.gap = nn.AdaptiveAvgPool1d(1)

        # ── Classification head ───────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(num_filters * 2, num_filters),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_filters, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 14)
        x = self.conv_block1(x)   # (B, num_filters,   14)
        x = self.conv_block2(x)   # (B, num_filters*2, 14)
        x = self.conv_block3(x)   # (B, num_filters*2, 14)
        x = self.gap(x)           # (B, num_filters*2,  1)
        x = x.squeeze(-1)         # (B, num_filters*2)
        return self.classifier(x) # (B, num_classes)
