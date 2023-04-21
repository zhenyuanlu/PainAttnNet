"""
module_module_mscn.py

This module contains the implementation of the Multiscale Convolutional Network (MSCN) architecture.
It provides the MSCN class.
"""
import torch
import torch.nn as nn


class MSCN(nn.Module):
    def __init__(self):
        super(MSCN, self).__init__()
        dropout = 0.5

        self.short_scale = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=6, dilation=1, bias=False, padding=24),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.Conv1d(128, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.MaxPool1d(kernel_size=4, stride=4, padding=2)
        )

        self.medium_scale = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=512, stride=42, dilation=1, bias=False, padding=256),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=0),
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=4, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.Conv1d(128, 64, kernel_size=4, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.long_scale = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=1024, stride=84, dilation=1, bias=False, padding=512),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=8, stride=8, padding=0),
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.Conv1d(128, 64, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.dropout = nn.Dropout(dropout)
        self.fc_short = nn.Linear(60, 75)
        self.fc_medium = nn.Linear(12, 75)
        self.fc_long = nn.Linear(3, 75)

    def forward(self, x):
        x_short = self.short_scale(x)
        # print(x_short.shape[2]) # (128, 64, 60) for BioVid
        x_medium = self.medium_scale(x)
        # print(x_medium.shape[2]) # (128, 64, 12) for BioVid
        x_long = self.long_scale(x)
        # print(x_long.shape[2]) # (128, 64, 3) for BioVid

        x_short = self.fc_short(x_short)
        x_medium = self.fc_medium(x_medium)
        x_long = self.fc_long(x_long)

        x_concat = torch.cat((x_short, x_medium, x_long), dim=1)

        x_concat = self.dropout(x_concat)

        return x_concat


