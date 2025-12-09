import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import cv2
import numpy as np
import torchvision.models.video as models
from torch.cuda.amp import autocast, GradScaler

# --- ⚙️ Configuration Area (Scratch Version) ---
CSV_FILE = "annotations/train_set_labels.csv"
VIDEO_FOLDER = "train_set" 
BATCH_SIZE = 4              
ACCUMULATION_STEPS = 4      
RESIZE_H, RESIZE_W = 128, 128 
NUM_FRAMES = 16             
LEARNING_RATE = 0.01
LEARNING_RATE = 0.001
EPOCHS = 45                 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Video Dataset ---
class IndustrialVideoDataset(Dataset):
    def __init__(self, csv_path, video_dir):
        self.labels_df = pd.read_csv(csv_path, header=None)
        self.video_dir = video_dir
        
        unique_labels = sorted(self.labels_df.iloc[:, 1].unique())
        self.label_to_int = {name: i for i, name in enumerate(unique_labels)}
        self.num_classes = len(unique_labels)
        print(f"📊 [Scratch] Video Dataset: {len(self.labels_df)} samples, {self.num_classes} classes")

    def __len__(self):
        return len(self.labels_df)

    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.resize(frame, (RESIZE_W, RESIZE_H))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        finally:
            cap.release()
            
        if len(frames) == 0:
            return np.zeros((NUM_FRAMES, RESIZE_H, RESIZE_W, 3), dtype=np.uint8)

        # Uniformly sample 16 frames
        indices = np.linspace(0, len(frames) - 1, NUM_FRAMES).astype(int)
        sampled_frames = np.array([frames[i] for i in indices])
        return sampled_frames

    def __getitem__(self, idx):
        file_name_avi = self.labels_df.iloc[idx, 0]
        if not file_name_avi.endswith('.avi'): file_name_avi += '.avi'
        video_path = os.path.join(self.video_dir, file_name_avi)
        label = self.label_to_int[self.labels_df.iloc[idx, 1]]
        
        buffer = self.load_video(video_path)
        buffer = torch.FloatTensor(buffer).permute(3, 0, 1, 2)
        buffer = buffer / 255.0 
        
        # Even when training from scratch, using Kinetics mean/std normalization is usually safe.
        # Alternatively: use mean=0.5/std=0.5. To keep variables controlled, we keep the same normalization as pretrained settings.
        mean = torch.tensor([0.432, 0.394, 0.376]).view(3, 1, 1, 1)
        std = torch.tensor([0.228, 0.221, 0.217]).view(3, 1, 1, 1)
        buffer = (buffer - mean) / std

        return buffer, torch.tensor(label, dtype=torch.long)

# --- 2. Main Training Loop ---
if __name__ == "__main__":
    print(f"🚀 RGB Training from Scratch | Device: {device}")
    
    torch.cuda.empty_cache()
    train_dataset = IndustrialVideoDataset(CSV_FILE, VIDEO_FOLDER)
    
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 🧠 Core Difference: weights=None (no pretrained weights)
    print("🧠 Initializing R2Plus1D (Random Initialization)...")
    model = models.r2plus1d_18(weights=None) 
    
    model.fc = nn.Linear(model.fc.in_features, train_dataset.num_classes)
    model = model.to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
    scaler = GradScaler()

    print("🔥 Starting Scratch Training...")
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels) / ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            running_loss += loss.item() * ACCUMULATION_STEPS
            
            if (i+1) % 50 == 0:
                print(f"  Step {i+1}/{len(train_loader)} | Loss: {loss.item()*ACCUMULATION_STEPS:.4f}")
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with autocast():
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        scheduler.step(val_acc)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            # 💾 Save scratch version
            torch.save(model.state_dict(), "best_model_rgb_scratch.pth")
            print(f"  💾 New Best Accuracy: {best_acc:.2f}%")

    print(f"🏆 Final Scratch RGB Accuracy: {best_acc:.2f}%")
