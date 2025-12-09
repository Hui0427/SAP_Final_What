import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import glob
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torchvision.models.video as models
from torch.utils.data import Dataset, DataLoader

# =========================================================================
# ⚙️ 全局配置 (Scratch Version)
# =========================================================================
SKELETON_MODEL_PATH = "best_model_v3.pth"        # 骨架模型保持不变
RGB_MODEL_PATH = "best_model_rgb_scratch.pth"    # ⚠️ 加载刚才跑出来的无预训练模型

# 数据路径
TRAIN_VIDEO_DIR = "train_set"
TRAIN_SKELETON_DIR = "skeleton_data/train"
TEST_VIDEO_DIR = "test_set"
TEST_SKELETON_DIR = "skeleton_data/test"
CSV_FILE = "annotations/train_set_labels.csv"
OUTPUT_FILE = "test_set_labels_scratch.csv" 

BATCH_SIZE = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================================
# 🏗️ 模型定义
# =========================================================================
class LightweightCNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LightweightCNNLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.1))
        self.lstm = nn.LSTM(64, hidden_size, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size*2, num_heads=4, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        c_in = x.permute(0, 2, 1)
        c_out = self.cnn(c_in)
        lstm_in = c_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(lstm_in)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        pooled = torch.mean(attn_out, dim=1)
        return self.fc(pooled)

# =========================================================================
# 🛠️ 通用数据处理函数
# =========================================================================
def load_process_skeleton(npy_path):
    if not os.path.exists(npy_path): return torch.zeros((1, 100, 99))
    raw = np.load(npy_path)
    if raw.shape[0] == 0: return torch.zeros((1, 100, 99))
    frames = raw.shape[0]
    data = raw.reshape(frames, 33, 4)[:, :, :3]
    root = (data[:, 23, :] + data[:, 24, :]) / 2
    data = data - root.reshape(frames, 1, 3)
    ls, rs = data[:, 11, :], data[:, 12, :]
    dist = np.sqrt(np.sum((ls - rs)**2, axis=1)).reshape(frames, 1, 1)
    dist = np.where(dist < 1e-4, 1.0, dist)
    data = (data / dist).reshape(frames, 99)
    if data.shape[0] > 100: 
        start = (data.shape[0]-100)//2
        data = data[start:start+100]
    elif data.shape[0] < 100: 
        data = np.vstack((np.zeros((100-data.shape[0], 99)), data))
    return torch.FloatTensor(data).unsqueeze(0)

def load_process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (128, 128))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    finally:
        cap.release()
    if len(frames) == 0: return torch.zeros((1, 3, 16, 128, 128))
    indices = np.linspace(0, len(frames)-1, 16).astype(int)
    buffer = torch.FloatTensor(np.array([frames[i] for i in indices])).permute(3, 0, 1, 2) / 255.0
    mean = torch.tensor([0.432, 0.394, 0.376]).view(3, 1, 1, 1)
    std = torch.tensor([0.228, 0.221, 0.217]).view(3, 1, 1, 1)
    buffer = (buffer - mean) / std
    return buffer.unsqueeze(0)

# =========================================================================
# 📊 验证与对比
# =========================================================================
class ValDataset(Dataset):
    def __init__(self, csv_path, vid_dir, skel_dir):
        self.labels_df = pd.read_csv(csv_path, header=None)
        self.vid_dir = vid_dir
        self.skel_dir = skel_dir
        self.unique_labels = sorted(self.labels_df.iloc[:, 1].unique())
        self.label_to_int = {name: i for i, name in enumerate(self.unique_labels)}
    def __len__(self): return len(self.labels_df)
    def __getitem__(self, idx):
        fid = os.path.splitext(self.labels_df.iloc[idx, 0])[0]
        lbl = self.label_to_int[self.labels_df.iloc[idx, 1]]
        skel = load_process_skeleton(os.path.join(self.skel_dir, fid+".npy")).squeeze(0)
        vid_path = os.path.join(self.vid_dir, fid+".avi")
        if not os.path.exists(vid_path): vid_path = os.path.join(self.vid_dir, fid+".mp4")
        rgb = load_process_video(vid_path).squeeze(0)
        return skel, rgb, lbl

def run_validation_and_search(skel_model, rgb_model):
    print(f"\n📊 [Scratch Comparison] 启动无预训练版本对比验证...")
    full_dataset = ValDataset(CSV_FILE, TRAIN_VIDEO_DIR, TRAIN_SKELETON_DIR)
    
    indices = list(range(len(full_dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(0.8 * len(full_dataset))
    val_indices = indices[split:]
    
    val_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, 
                            sampler=torch.utils.data.SubsetRandomSampler(val_indices), num_workers=0)
    
    print(f"📉 验证集大小: {len(val_indices)}")
    
    all_skel_probs, all_rgb_probs, all_labels = [], [], []
    
    with torch.no_grad():
        for i, (skel, rgb, lbl) in enumerate(val_loader):
            skel, rgb, lbl = skel.to(device), rgb.to(device), lbl.to(device)
            s_out = torch.softmax(skel_model(skel), dim=1)
            r_out = torch.softmax(rgb_model(rgb), dim=1)
            all_skel_probs.append(s_out.cpu())
            all_rgb_probs.append(r_out.cpu())
            all_labels.append(lbl.cpu())
            if (i+1)%10 == 0: print(f"  Batch {i+1} done...", end="\r")
            
    final_skel = torch.cat(all_skel_probs)
    final_rgb = torch.cat(all_rgb_probs)
    final_lbl = torch.cat(all_labels)
    
    # 1. 计算单模态准确率
    skel_acc = (final_skel.argmax(1) == final_lbl).float().mean().item() * 100
    rgb_acc = (final_rgb.argmax(1) == final_lbl).float().mean().item() * 100
    
    # 2. 网格搜索最佳权重
    best_acc = 0.0
    best_alpha = 0.0
    
    for alpha in np.linspace(0, 1, 101):
        fusion = (alpha * final_rgb) + ((1 - alpha) * final_skel)
        acc = (fusion.argmax(1) == final_lbl).float().mean().item() * 100
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha
            
    print("\n" + "="*60)
    print("🏆 [Scratch] 无预训练版本性能对比 (Hypothesis Verification)")
    print("="*60)
    print(f"🦴 单骨架流 (Skeleton): {skel_acc:.2f}% (Baseline)")
    print(f"🎨 单RGB流 (Scratch):  {rgb_acc:.2f}% (Expect Low)")
    print("-" * 60)
    print(f"🚀 双流融合 (Fusion):   {best_acc:.2f}%")
    print(f"⚖️ 权重变化: RGB={best_alpha:.2f}, Skeleton={1-best_alpha:.2f}")
    print("   (预期：Skeleton 权重应大幅增加)")
    print("="*60)
    
    # 生成图表
    best_fusion = (best_alpha * final_rgb) + ((1 - best_alpha) * final_skel)
    cm = confusion_matrix(final_lbl.numpy(), best_fusion.argmax(1).numpy())
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap='Reds', # 用红色系区分
                xticklabels=full_dataset.unique_labels, yticklabels=full_dataset.unique_labels)
    plt.title(f'Scratch Fusion Matrix (Acc: {best_acc:.2f}%)')
    plt.xticks(rotation=90); plt.tight_layout()
    plt.savefig('fusion_analysis_scratch.png')
    print("✅ 验证图表已保存: fusion_analysis_scratch.png")

if __name__ == "__main__":
    df = pd.read_csv(CSV_FILE, header=None)
    num_classes = len(df.iloc[:, 1].unique())
    
    # 加载骨架 (V3)
    skel_model = LightweightCNNLSTM(99, 128, num_classes).to(device)
    skel_model.load_state_dict(torch.load(SKELETON_MODEL_PATH, map_location=device))
    skel_model.eval()
    
    # 加载 Scratch RGB
    print(f"🧠 加载 Scratch RGB 模型: {RGB_MODEL_PATH}")
    if not os.path.exists(RGB_MODEL_PATH):
        print("❌ 错误: 找不到 scratch 模型，请先运行 train_rgb_scratch.py")
        exit()
        
    rgb_model = models.r2plus1d_18(weights=None) # ⚠️ 关键：weights=None
    rgb_model.fc = nn.Linear(rgb_model.fc.in_features, num_classes)
    rgb_model.load_state_dict(torch.load(RGB_MODEL_PATH, map_location=device))
    rgb_model.to(device)
    rgb_model.eval()
    
    run_validation_and_search(skel_model, rgb_model)