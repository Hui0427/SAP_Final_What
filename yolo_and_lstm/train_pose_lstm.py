import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd  # 文件最上面加这一行

from skeleton_lstm_dataset import SkeletonSequenceDataset
from skeleton_lstm_model import PoseLSTM

META_CSV   = "keypoints/meta.csv"
SEQ_LEN    = 64
BATCH_SIZE = 16
EPOCHS     = 30
LR         = 1e-3

# 从 meta.csv 自动读取类别数量，并做 sanity check
meta_df = pd.read_csv(META_CSV)
min_label = meta_df["label_id"].min()
max_label = meta_df["label_id"].max()
unique_labels = sorted(meta_df["label_id"].unique())
NUM_CLASSES = max_label + 1

print("标签统计：")
print("  unique labels:", unique_labels)
print("  min label:", min_label)
print("  max label:", max_label)
print("  NUM_CLASSES (自动计算):", NUM_CLASSES)

assert min_label >= 0, "label_id 不能有负数！"
assert max_label == NUM_CLASSES - 1, "label_id 必须是从 0 到 NUM_CLASSES-1 的连续整数！"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for seqs, labels in tqdm(loader, desc="Train", leave=False):
        seqs = seqs.to(DEVICE)           # [B, L, D]
        labels = labels.to(DEVICE)       # [B]

        optimizer.zero_grad()
        logits = model(seqs)             # [B, C]
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * seqs.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += seqs.size(0)

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return avg_loss, acc

def eval_one_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for seqs, labels in tqdm(loader, desc="Eval", leave=False):
            seqs = seqs.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(seqs)
            loss = criterion(logits, labels)

            total_loss += loss.item() * seqs.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += seqs.size(0)

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return avg_loss, acc

def main():
    # 先用一个 dataset 看一下 input_dim
    tmp_dataset = SkeletonSequenceDataset(META_CSV, seq_len=SEQ_LEN, split="train")
    tmp_seq, _ = tmp_dataset[0]
    input_dim = tmp_seq.shape[1]
    print("Input dim:", input_dim)

    # 重新构造 train/val
    train_dataset = SkeletonSequenceDataset(META_CSV, seq_len=SEQ_LEN, split="train")
    val_dataset   = SkeletonSequenceDataset(META_CSV, seq_len=SEQ_LEN, split="val")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = PoseLSTM(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=2,
        num_classes=NUM_CLASSES,
        bidirectional=True,
        dropout=0.3,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion)

        print(f"  Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        print(f"  Val   loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = f"checkpoints/pose_lstm_best.pt"
            torch.save({
                "model_state": model.state_dict(),
                "input_dim": input_dim,
                "num_classes": NUM_CLASSES,
                "seq_len": SEQ_LEN,
            }, ckpt_path)
            print(f"  Saved best model to {ckpt_path}")

if __name__ == "__main__":
    main()
