import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import torchvision.models.video as models
from torch.utils.data import Dataset, DataLoader

# =========================================================================
# ⚙️ Configuration
# =========================================================================
SKELETON_MODEL_PATH = "best_model_v3.pth"
RGB_MODEL_PATH = "best_model_rgb.pth"

# NOTE: We use the training set here to simulate validation
TRAIN_VIDEO_DIR = "train_set"
TRAIN_SKELETON_DIR = "skeleton_data/train"
CSV_FILE = "annotations/train_set_labels.csv"

# Fusion weights
ALPHA_RGB = 0.8
ALPHA_SKELETON = 0.2

BATCH_SIZE = 8  # No backprop during validation, can be slightly larger
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================================
# 🏗️ Model Definition & Data Processing (same logic as training)
# =========================================================================

# --- 1. Skeleton Model ---
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

# --- 2. Dual-Stream Dataset ---
class DualStreamDataset(Dataset):
    def __init__(self, csv_path, video_dir, skeleton_dir):
        self.labels_df = pd.read_csv(csv_path, header=None)
        self.video_dir = video_dir
        self.skeleton_dir = skeleton_dir
        self.unique_labels = sorted(self.labels_df.iloc[:, 1].unique())
        self.label_to_int = {name: i for i, name in enumerate(self.unique_labels)}

    def __len__(self):
        return len(self.labels_df)

    def load_skeleton(self, file_id):
        # Skeleton preprocessing logic (same as train_model_v3.py)
        path = os.path.join(self.skeleton_dir, file_id + ".npy")
        if not os.path.exists(path): return torch.zeros((100, 99))
        raw = np.load(path)
        if raw.shape[0] == 0: return torch.zeros((100, 99))
        
        # Normalize
        frames = raw.shape[0]
        data = raw.reshape(frames, 33, 4)[:, :, :3]
        root = (data[:, 23, :] + data[:, 24, :]) / 2
        data = data - root.reshape(frames, 1, 3)
        ls, rs = data[:, 11, :], data[:, 12, :]
        dist = np.sqrt(np.sum((ls - rs)**2, axis=1)).reshape(frames, 1, 1)
        dist = np.where(dist < 1e-4, 1.0, dist)
        data = (data / dist).reshape(frames, 99)
        
        # Pad
        if data.shape[0] > 100: data = data[(data.shape[0]-100)//2 : (data.shape[0]-100)//2+100]
        elif data.shape[0] < 100: data = np.vstack((np.zeros((100-data.shape[0], 99)), data))
        
        return torch.FloatTensor(data)

    def load_video(self, file_id):
        # RGB processing logic (same as rgb_model.py)
        path = os.path.join(self.video_dir, file_id + ".avi")
        cap = cv2.VideoCapture(path)
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
            
        if len(frames) == 0: return torch.zeros((3, 16, 128, 128))
        
        indices = np.linspace(0, len(frames)-1, 16).astype(int)
        buffer = torch.FloatTensor(np.array([frames[i] for i in indices])).permute(3, 0, 1, 2) / 255.0
        mean = torch.tensor([0.432, 0.394, 0.376]).view(3, 1, 1, 1)
        std = torch.tensor([0.228, 0.221, 0.217]).view(3, 1, 1, 1)
        return (buffer - mean) / std

    def __getitem__(self, idx):
        fname = self.labels_df.iloc[idx, 0]
        fid = os.path.splitext(fname)[0]
        label = self.label_to_int[self.labels_df.iloc[idx, 1]]
        
        skel = self.load_skeleton(fid)
        rgb = self.load_video(fid)
        
        return skel, rgb, label

# =========================================================================
# 🚀 Main Program: Validate Fusion Accuracy
# =========================================================================
if __name__ == "__main__":
    print(f"📊 Starting local validation | Device: {device}")
    
    # 1. Create validation split (random 20% of full dataset)
    full_dataset = DualStreamDataset(CSV_FILE, TRAIN_VIDEO_DIR, TRAIN_SKELETON_DIR)
    dataset_len = len(full_dataset)
    indices = list(range(dataset_len))
    
    # Shuffle for balanced validation
    np.random.seed(42)
    np.random.shuffle(indices)
    
    split = int(0.8 * dataset_len)
    val_indices = indices[split:]
    
    val_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, 
                            sampler=torch.utils.data.SubsetRandomSampler(val_indices),
                            num_workers=0)
    
    num_classes = len(full_dataset.unique_labels)
    print(f"📉 Validation set size: {len(val_indices)} samples")

    # 2. Load models
    print("🧠 Loading models...")
    # Skeleton
    skel_model = LightweightCNNLSTM(99, 128, num_classes).to(device)
    skel_model.load_state_dict(torch.load(SKELETON_MODEL_PATH, map_location=device))
    skel_model.eval()
    
    # RGB
    rgb_model = models.r2plus1d_18(weights=None)
    rgb_model.fc = nn.Linear(rgb_model.fc.in_features, num_classes)
    rgb_model.load_state_dict(torch.load(RGB_MODEL_PATH, map_location=device))
    rgb_model.to(device)
    rgb_model.eval()

    # 3. Start inference
    print("🔥 Running fusion inference...")
    all_preds = []
    all_labels = []
    
    rgb_correct = 0
    skel_correct = 0
    fusion_correct = 0
    total = 0
    
    with torch.no_grad():
        for i, (skel_in, rgb_in, labels) in enumerate(val_loader):
            skel_in, rgb_in, labels = skel_in.to(device), rgb_in.to(device), labels.to(device)

            # Separate predictions
            skel_out = torch.softmax(skel_model(skel_in), dim=1)
            rgb_out = torch.softmax(rgb_model(rgb_in), dim=1)

            # Fusion
            fusion_out = (ALPHA_RGB * rgb_out) + (ALPHA_SKELETON * skel_out)

            # Statistics
            _, skel_pred = torch.max(skel_out, 1)
            _, rgb_pred = torch.max(rgb_out, 1)
            _, fusion_pred = torch.max(fusion_out, 1)
            
            skel_correct += (skel_pred == labels).sum().item()
            rgb_correct += (rgb_pred == labels).sum().item()
            fusion_correct += (fusion_pred == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(fusion_pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if (i+1) % 10 == 0: 
                print(f"  Batch {i+1} done...")

    # 4. Print results
    print("-" * 40)
    print(f"🏆 Validation Results (Val Set N={total})")
    print("-" * 40)
    print(f"🦴 Skeleton-only accuracy: {100 * skel_correct / total:.2f}%")
    print(f"🎨 RGB-only accuracy:      {100 * rgb_correct / total:.2f}%")
    print(f"🚀 Fusion accuracy:         {100 * fusion_correct / total:.2f}%")
    print("-" * 40)
    
    # 5. Plot matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap='Blues', 
                xticklabels=full_dataset.unique_labels, 
                yticklabels=full_dataset.unique_labels)
    plt.title(f'Fusion Model Confusion Matrix (Acc: {100 * fusion_correct / total:.2f}%)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('fusion_analysis.png')
    print("✅ Fusion confusion matrix saved as fusion_analysis.png")

