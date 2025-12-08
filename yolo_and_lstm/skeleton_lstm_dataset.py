import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class SkeletonSequenceDataset(Dataset):
    def __init__(self, meta_csv, seq_len=64, split="train", split_ratio=(0.7, 0.15, 0.15), seed=42):
        """
        meta_csv: keypoints/meta.csv
        split: "train" / "val" / "test"
        """
        self.seq_len = seq_len
        df = pd.read_csv(meta_csv)

        # 固定随机种子，划分 train/val/test
        rng = np.random.RandomState(seed)
        indices = np.arange(len(df))
        rng.shuffle(indices)

        n = len(df)
        n_train = int(n * split_ratio[0])
        n_val = int(n * split_ratio[1])
        train_idx = indices[:n_train]
        val_idx   = indices[n_train:n_train + n_val]
        test_idx  = indices[n_train + n_val:]

        if split == "train":
            self.df = df.iloc[train_idx].reset_index(drop=True)
        elif split == "val":
            self.df = df.iloc[val_idx].reset_index(drop=True)
        else:
            self.df = df.iloc[test_idx].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _resample_seq(self, seq):
        """把 [T, D] 的序列变成固定长度 [seq_len, D]"""
        T, D = seq.shape
        if T == self.seq_len:
            return seq
        if T > self.seq_len:
            # 均匀采样 seq_len 帧
            idxs = np.linspace(0, T - 1, self.seq_len).astype(int)
            return seq[idxs]
        else:
            # T < seq_len，后面补零
            out = np.zeros((self.seq_len, D), dtype=np.float32)
            out[:T] = seq
            return out

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["npy_path"]
        label = int(row["label_id"])

        seq = np.load(path).astype(np.float32)  # [T, D]
        seq = self._resample_seq(seq)          # [L, D]

        # 转成 torch tensor，形状 [L, D]
        seq_tensor = torch.from_numpy(seq)     # float32
        label_tensor = torch.tensor(label, dtype=torch.long)
        return seq_tensor, label_tensor
