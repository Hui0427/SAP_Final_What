import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import random
from sklearn.utils.class_weight import compute_class_weight

# --- ⚙️ Configuration Area ---
CSV_FILE = "annotations/train_set_labels.csv"
DATA_FOLDER = "skeleton_data/train"
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 80
FIXED_LENGTH = 100
INPUT_SIZE = 99     # Pure XYZ (33*3), removing angle features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Dataset (Keep good normalization from V4/V7, remove problematic angle features) ---
class SkeletonDatasetV8(Dataset):
    def __init__(self, csv_path, data_dir, sequence_length, augment=False):
        self.labels_df = pd.read_csv(csv_path, header=None)
        self.data_dir = data_dir
        self.seq_len = sequence_length
        self.augment = augment
        
        unique_labels = sorted(self.labels_df.iloc[:, 1].unique())
        self.label_to_int = {name: i for i, name in enumerate(unique_labels)}
        self.all_labels = [self.label_to_int[label] for label in self.labels_df.iloc[:, 1]]

    def __len__(self):
        return len(self.labels_df)

    def normalize_skeleton(self, raw_data):
        frames = raw_data.shape[0]
        data = raw_data.reshape(frames, 33, 4)
        xyz = data[:, :, :3]
        
        # 1. Root Centering (Position Normalization)
        root = (xyz[:, 23, :] + xyz[:, 24, :]) / 2
        xyz = xyz - root.reshape(frames, 1, 3)
        
        # 2. Shoulder Scaling (Scale Normalization)
        left_shoulder = xyz[:, 11, :]
        right_shoulder = xyz[:, 12, :]
        dist = np.sqrt(np.sum((left_shoulder - right_shoulder)**2, axis=1))
        dist = np.where(dist < 1e-4, 1.0, dist).reshape(frames, 1, 1)
        xyz_norm = xyz / dist
        
        return xyz_norm.reshape(frames, 99)

    def apply_augmentation(self, data):
        # Lightweight augmentation to reduce overfitting
        if random.random() > 0.5:  # Add noise
            data = data + np.random.normal(0, 0.002, data.shape)
            
        if random.random() > 0.6:  # Random occlusion
            temp = data.reshape(-1, 33, 3)
            # Randomly zero out one hand
            if random.random() > 0.5:  # Left hand
                temp[:, [11,13,15], :] = 0
            else:  # Right hand
                temp[:, [12,14,16], :] = 0
            data = temp.reshape(-1, 99)
            
        return data

    def __getitem__(self, idx):
        file_name_avi = self.labels_df.iloc[idx, 0]
        file_id = os.path.splitext(file_name_avi)[0]
        npy_path = os.path.join(self.data_dir, file_id + ".npy")
        label = self.label_to_int[self.labels_df.iloc[idx, 1]]
        
        if os.path.exists(npy_path):
            raw_data = np.load(npy_path)
            data = self.normalize_skeleton(raw_data) if raw_data.shape[0]>0 else np.zeros((self.seq_len, 99))
        else:
            data = np.zeros((self.seq_len, 99))

        # Padding/Truncating
        current_len = data.shape[0]
        if current_len > self.seq_len:
            start = (current_len - self.seq_len) // 2
            data = data[start : start + self.seq_len, :]
        elif current_len < self.seq_len:
            padding = np.zeros((self.seq_len - current_len, 99))
            data = np.vstack((padding, data))

        if self.augment:
            data = self.apply_augmentation(data)

        return torch.FloatTensor(data), torch.tensor(label, dtype=torch.long)

# --- 2. Lightweight Model (Lightweight CNN-LSTM) ---
class LightweightCNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LightweightCNNLSTM, self).__init__()
        
        # 1. Shallow CNN: Used only for local feature extraction, does not change temporal length
        # Keep kernel_size=3, padding=1 so seq_len remains unchanged
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1) 
            # ❌ Removed Pooling layer to preserve full temporal resolution
        )
        
        # 2. LSTM: Handles temporal modeling (hidden_size 128 is sufficient; no need for 256)
        # Input dimension is CNN output = 64
        self.lstm = nn.LSTM(64, hidden_size, num_layers=2, 
                            batch_first=True, bidirectional=True, dropout=0.3)
        
        # 3. Attention
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size*2, num_heads=4, batch_first=True)
        
        # 4. Classifier
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x: [batch, seq, 99]
        
        # CNN
        c_in = x.permute(0, 2, 1)  # -> [batch, 99, seq]
        c_out = self.cnn(c_in)     # -> [batch, 64, seq]
        lstm_in = c_out.permute(0, 2, 1)  # -> [batch, seq, 64]
        
        # LSTM
        lstm_out, _ = self.lstm(lstm_in)
        
        # Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Mean Pooling
        pooled = torch.mean(attn_out, dim=1)
        
        out = self.fc(pooled)
        return out

# --- 3. Training Controller ---
if __name__ == "__main__":
    print(f"🚀 V8 Slim Version Start | Removed redundant parameters, back to essentials | Device: {device}")
    
    train_full = SkeletonDatasetV8(CSV_FILE, DATA_FOLDER, FIXED_LENGTH, augment=True)
    val_full = SkeletonDatasetV8(CSV_FILE, DATA_FOLDER, FIXED_LENGTH, augment=False)
    
    # Class Weights
    y_train = train_full.all_labels
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Dataloaders
    dataset_len = len(train_full)
    indices = list(range(dataset_len))
    split = int(0.2 * dataset_len)
    np.random.shuffle(indices)
    train_idx, val_idx = indices[split:], indices[:split]
    
    train_loader = DataLoader(train_full, batch_size=BATCH_SIZE, sampler=torch.utils.data.SubsetRandomSampler(train_idx))
    val_loader = DataLoader(val_full, batch_size=BATCH_SIZE, sampler=torch.utils.data.SubsetRandomSampler(val_idx))
    
    # Model Init
    num_classes = len(train_full.label_to_int)
    model = LightweightCNNLSTM(input_size=INPUT_SIZE, hidden_size=128, num_classes=num_classes).to(device)
    
    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📉 Model Parameter Count: {total_params:,} (Previous V7 was 3.2 million)")
    
    # Stable Scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    print("🔥 Training Starts...")
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        
        # Scheduler Step
        scheduler.step(val_acc)
        lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {lr:.6f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model_v3.pth")
            print(f"  💾 New Best Score! (Best: {best_acc:.2f}%)")

    print(f"\n🏆 V8 Final Result: {best_acc:.2f}%")
