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
# ⚙️ Global Configuration
# =========================================================================
SKELETON_MODEL_PATH = "best_model_v3.pth"
RGB_MODEL_PATH = "best_model_rgb.pth"

# Data paths
TRAIN_VIDEO_DIR = "train_set"
TRAIN_SKELETON_DIR = "skeleton_data/train"
TEST_VIDEO_DIR = "test_set"
TEST_SKELETON_DIR = "skeleton_data/test"
CSV_FILE = "annotations/train_set_labels.csv"  # Must exist for loading ID mappings
OUTPUT_FILE = "test_set_labels.csv"

BATCH_SIZE = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================================
# 🏗️ Model Definition
# =========================================================================
class LightweightCNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LightweightCNNLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.1))
        self.lstm = nn.LSTM(64, hidden_size, num_layers=2, batch_first=True, 
                            bidirectional=True, dropout=0.3)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size*2, 
                                               num_heads=4, batch_first=True)
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
# 🛠️ Data Processing Functions
# =========================================================================
def load_process_skeleton(npy_path):
    if not os.path.exists(npy_path):
        return torch.zeros((1, 100, 99))
    raw = np.load(npy_path)
    if raw.shape[0] == 0:
        return torch.zeros((1, 100, 99))

    frames = raw.shape[0]
    data = raw.reshape(frames, 33, 4)[:, :, :3]

    # Root-centering
    root = (data[:, 23, :] + data[:, 24, :]) / 2
    data = data - root.reshape(frames, 1, 3)

    # Shoulder scaling
    ls, rs = data[:, 11, :], data[:, 12, :]
    dist = np.sqrt(np.sum((ls - rs)**2, axis=1)).reshape(frames, 1, 1)
    dist = np.where(dist < 1e-4, 1.0, dist)
    data = (data / dist).reshape(frames, 99)

    # Trim or pad to 100 frames
    if data.shape[0] > 100:
        start = (data.shape[0] - 100) // 2
        data = data[start:start+100]
    elif data.shape[0] < 100:
        data = np.vstack((np.zeros((100 - data.shape[0], 99)), data))

    return torch.FloatTensor(data).unsqueeze(0)

def load_process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (128, 128))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    finally:
        cap.release()

    if len(frames) == 0:
        return torch.zeros((1, 3, 16, 128, 128))

    indices = np.linspace(0, len(frames)-1, 16).astype(int)
    buffer = torch.FloatTensor(np.array([frames[i] for i in indices])).permute(3, 0, 1, 2) / 255.0

    mean = torch.tensor([0.432, 0.394, 0.376]).view(3, 1, 1, 1)
    std = torch.tensor([0.228, 0.221, 0.217]).view(3, 1, 1, 1)
    buffer = (buffer - mean) / std

    return buffer.unsqueeze(0)

# =========================================================================
# 📊 Phase 1: Validation and Weight Search
# =========================================================================
class ValDataset(Dataset):
    def __init__(self, csv_path, vid_dir, skel_dir):
        self.labels_df = pd.read_csv(csv_path, header=None)
        self.vid_dir = vid_dir
        self.skel_dir = skel_dir
        self.unique_labels = sorted(self.labels_df.iloc[:, 1].unique())
        self.label_to_int = {name: i for i, name in enumerate(self.unique_labels)}

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        fid = os.path.splitext(self.labels_df.iloc[idx, 0])[0]
        lbl = self.label_to_int[self.labels_df.iloc[idx, 1]]

        skel = load_process_skeleton(os.path.join(self.skel_dir, fid + ".npy")).squeeze(0)

        vid_path = os.path.join(self.vid_dir, fid + ".avi")
        if not os.path.exists(vid_path):
            vid_path = os.path.join(self.vid_dir, fid + ".mp4")

        rgb = load_process_video(vid_path).squeeze(0)

        return skel, rgb, lbl


def run_validation_and_search(skel_model, rgb_model):
    print(f"\n📊 [Phase 1] Starting validation weight search...")
    full_dataset = ValDataset(CSV_FILE, TRAIN_VIDEO_DIR, TRAIN_SKELETON_DIR)

    indices = list(range(len(full_dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(0.8 * len(full_dataset))
    val_indices = indices[split:]

    val_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE,
                            sampler=torch.utils.data.SubsetRandomSampler(val_indices),
                            num_workers=0)

    all_skel_probs, all_rgb_probs, all_labels = [], [], []

    with torch.no_grad():
        for i, (skel, rgb, lbl) in enumerate(val_loader):
            skel, rgb, lbl = skel.to(device), rgb.to(device), lbl.to(device)

            s_out = torch.softmax(skel_model(skel), dim=1)
            r_out = torch.softmax(rgb_model(rgb), dim=1)

            all_skel_probs.append(s_out.cpu())
            all_rgb_probs.append(r_out.cpu())
            all_labels.append(lbl.cpu())

            if (i+1) % 10 == 0:
                print(f"  Batch {i+1} done...", end="\r")

    final_skel = torch.cat(all_skel_probs)
    final_rgb = torch.cat(all_rgb_probs)
    final_lbl = torch.cat(all_labels)

    best_acc = 0.0
    best_alpha = 0.0

    for alpha in np.linspace(0, 1, 101):
        fusion = (alpha * final_rgb) + ((1 - alpha) * final_skel)
        acc = (fusion.argmax(1) == final_lbl).float().mean().item() * 100

        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha

    print(f"\n🚀 Best validation result: {best_acc:.2f}% (Alpha={best_alpha:.2f})")

    best_fusion = (best_alpha * final_rgb) + ((1 - best_alpha) * final_skel)
    cm = confusion_matrix(final_lbl.numpy(), best_fusion.argmax(1).numpy())

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap='Blues',
                xticklabels=full_dataset.unique_labels,
                yticklabels=full_dataset.unique_labels)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('fusion_analysis_final.png')

    return best_alpha

# =========================================================================
# 🚀 Phase 2: Generate Submission File (3-column format)
# =========================================================================
def generate_submission(skel_model, rgb_model, best_alpha, int_to_label):
    print(f"\n📦 [Phase 2] Generating 3-column submission file...")

    # 1. Build Label -> ClassID mapping
    train_df = pd.read_csv(CSV_FILE, header=None)
    label_to_id_map = dict(zip(train_df[1], train_df[2]))

    print(f"📋 Loaded {len(label_to_id_map)} class ID mappings")

    # 2. Collect and sort test files
    test_files = sorted(glob.glob(os.path.join(TEST_VIDEO_DIR, "*.avi")) +
                        glob.glob(os.path.join(TEST_VIDEO_DIR, "*.mp4")))
    print(f"📄 Found {len(test_files)} test videos")

    results = []
    with torch.no_grad():
        for i, vid_path in enumerate(test_files):
            fid = os.path.splitext(os.path.basename(vid_path))[0]
            npy_path = os.path.join(TEST_SKELETON_DIR, fid + ".npy")

            skel_in = load_process_skeleton(npy_path).to(device)
            rgb_in = load_process_video(vid_path).to(device)

            s_prob = torch.softmax(skel_model(skel_in), dim=1)
            r_prob = torch.softmax(rgb_model(rgb_in), dim=1)

            fusion_prob = (best_alpha * r_prob) + ((1 - best_alpha) * s_prob)
            pred_idx = torch.max(fusion_prob, 1)[1].item()

            label_name = int_to_label[pred_idx]

            class_id = label_to_id_map[label_name]

            results.append([fid + ".avi", label_name, class_id])

            if (i+1) % 50 == 0:
                print(f"  Processed {i+1}/{len(test_files)}", end="\r")

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False, header=False)

    print(f"\n\n🎉 Submission file generated: {OUTPUT_FILE}")
    print("💡 Example: testvideo001.avi, Walking, 27")

if __name__ == "__main__":
    # Load label list
    df = pd.read_csv(CSV_FILE, header=None)
    unique_labels = sorted(df.iloc[:, 1].unique())
    num_classes = len(unique_labels)
    int_to_label = {i: name for i, name in enumerate(unique_labels)}

    # Load models
    print("🧠 Loading model weights...")
    skel_model = LightweightCNNLSTM(99, 128, num_classes).to(device)
    skel_model.load_state_dict(torch.load(SKELETON_MODEL_PATH, map_location=device))
    skel_model.eval()

    rgb_model = models.r2plus1d_18(weights=None)
    rgb_model.fc = nn.Linear(rgb_model.fc.in_features, num_classes)
    rgb_model.load_state_dict(torch.load(RGB_MODEL_PATH, map_location=device))
    rgb_model.to(device)
    rgb_model.eval()

    # Execute pipeline
    best_alpha = run_validation_and_search(skel_model, rgb_model)
    generate_submission(skel_model, rgb_model, best_alpha, int_to_label)
