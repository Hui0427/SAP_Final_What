import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import glob
import cv2
import torchvision.models.video as models

# =========================================================================
# ⚙️ 配置区域 (请确认文件名是否与你实际保存的一致)
# =========================================================================
# 模型权重路径
SKELETON_MODEL_PATH = "best_model_v3.pth"   # 对应 train_model_v3.py 训练出的模型
RGB_MODEL_PATH = "best_model_rgb.pth"       # 对应 rgb_model.py 训练出的模型

# 数据路径
TEST_SKELETON_DIR = "skeleton_data/test"    # batch_extract.py 生成的骨架文件夹
TEST_VIDEO_DIR = "test_set"                 # 原始视频文件夹
TRAIN_CSV = "annotations/train_set_labels.csv" # 用于获取标签列表
OUTPUT_FILE = "test_set_labels_fusion.csv"  # 最终提交文件

# ⚖️ 融合权重 (关键策略：RGB为主，骨架为辅)
# RGB (91% Acc) 权重给高点；骨架 (70% Acc) 负责修正光照/遮挡带来的Corner Case
ALPHA_RGB = 0.8
ALPHA_SKELETON = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================================
# 🏗️ 1. 骨架模型架构 (必须完全复刻 train_model_v3.py)
# =========================================================================
class LightweightCNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LightweightCNNLSTM, self).__init__()
        
        # 1. 浅层 CNN
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1) 
        )
        
        # 2. LSTM
        self.lstm = nn.LSTM(64, hidden_size, num_layers=2, 
                            batch_first=True, bidirectional=True, dropout=0.3)
        
        # 3. Attention
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size*2, num_heads=4, batch_first=True)
        
        # 4. Classifier
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x: [batch, seq, 99]
        c_in = x.permute(0, 2, 1) # -> [batch, 99, seq]
        c_out = self.cnn(c_in)    # -> [batch, 64, seq]
        lstm_in = c_out.permute(0, 2, 1) # -> [batch, seq, 64]
        
        lstm_out, _ = self.lstm(lstm_in)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        pooled = torch.mean(attn_out, dim=1)
        out = self.fc(pooled)
        return out

# =========================================================================
# 🛠️ 2. 数据处理函数 (严格对应 batch_extract.py 和 rgb_model.py)
# =========================================================================

# --- A. 骨架处理 (train_model_v3.py 的逻辑) ---
def process_skeleton(npy_path):
    FIXED_LENGTH = 100
    
    # 容错处理：如果文件不存在或为空，返回全0张量
    if not os.path.exists(npy_path): return torch.zeros((1, FIXED_LENGTH, 99))
    raw_data = np.load(npy_path)
    if raw_data.shape[0] == 0: return torch.zeros((1, FIXED_LENGTH, 99))

    # 1. Reshape & Slice (132 -> 33*4 -> 33*3)
    # train_model_v3.py 只取了前3维 (x,y,z)，丢弃了 visibility
    frames = raw_data.shape[0]
    data = raw_data.reshape(frames, 33, 4)
    xyz = data[:, :, :3] 
    
    # 2. Root Centering (髋关节中心化)
    # 23=左髋, 24=右髋
    root = (xyz[:, 23, :] + xyz[:, 24, :]) / 2
    xyz = xyz - root.reshape(frames, 1, 3)
    
    # 3. Shoulder Scaling (肩宽归一化)
    # 11=左肩, 12=右肩
    left_shoulder = xyz[:, 11, :]
    right_shoulder = xyz[:, 12, :]
    dist = np.sqrt(np.sum((left_shoulder - right_shoulder)**2, axis=1))
    dist = np.where(dist < 1e-4, 1.0, dist).reshape(frames, 1, 1)
    xyz_norm = xyz / dist
    
    data = xyz_norm.reshape(frames, 99)

    # 4. Padding / Truncating (固定长度 100)
    if data.shape[0] > FIXED_LENGTH:
        start = (data.shape[0] - FIXED_LENGTH) // 2
        data = data[start : start + FIXED_LENGTH, :]
    elif data.shape[0] < FIXED_LENGTH:
        padding = np.zeros((FIXED_LENGTH - data.shape[0], 99))
        data = np.vstack((padding, data))
        
    return torch.FloatTensor(data).unsqueeze(0) # 增加 batch 维度 -> [1, 100, 99]

# --- B. RGB 视频处理 (rgb_model.py 的逻辑) ---
def process_video(video_path):
    RESIZE_H, RESIZE_W = 128, 128
    NUM_FRAMES = 16
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            # 必须和训练时一致：Resize -> BGR转RGB
            frame = cv2.resize(frame, (RESIZE_W, RESIZE_H))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    finally:
        cap.release()
        
    if len(frames) == 0:
        return torch.zeros((1, 3, NUM_FRAMES, RESIZE_H, RESIZE_W))

    # 1. 均匀采样 16 帧
    indices = np.linspace(0, len(frames) - 1, NUM_FRAMES).astype(int)
    sampled_frames = np.array([frames[i] for i in indices])
    
    # 2. 转 Tensor & 归一化 (Kinetics-400 参数)
    buffer = torch.FloatTensor(sampled_frames).permute(3, 0, 1, 2) / 255.0
    
    # 手动归一化 (对应 rgb_model.py 中的修复代码)
    mean = torch.tensor([0.432, 0.394, 0.376]).view(3, 1, 1, 1)
    std = torch.tensor([0.228, 0.221, 0.217]).view(3, 1, 1, 1)
    buffer = (buffer - mean) / std
    
    return buffer.unsqueeze(0) # 增加 batch 维度 -> [1, 3, 16, 128, 128]

# =========================================================================
# 🚀 3. 主执行逻辑
# =========================================================================
if __name__ == "__main__":
    print(f"🚀 启动多模态融合推理 (Two-Stream Fusion) | 设备: {device}")
    
    # 1. 加载标签映射
    df = pd.read_csv(TRAIN_CSV, header=None)
    unique_labels = sorted(df.iloc[:, 1].unique())
    int_to_label = {i: name for i, name in enumerate(unique_labels)}
    num_classes = len(unique_labels)
    print(f"📋 标签加载完毕: {num_classes} 类")

    # 2. 加载骨架模型
    print(f"🧠 加载骨架模型: {SKELETON_MODEL_PATH}")
    if not os.path.exists(SKELETON_MODEL_PATH):
        print(f"❌ 错误: 找不到骨架模型文件 {SKELETON_MODEL_PATH}")
        exit()
    # 骨架模型参数必须与 train_model_v3.py 一致: input=99, hidden=128
    skel_model = LightweightCNNLSTM(input_size=99, hidden_size=128, num_classes=num_classes).to(device)
    skel_model.load_state_dict(torch.load(SKELETON_MODEL_PATH, map_location=device))
    skel_model.eval()

    # 3. 加载 RGB 模型
    print(f"🧠 加载 RGB 模型: {RGB_MODEL_PATH}")
    if not os.path.exists(RGB_MODEL_PATH):
        print(f"❌ 错误: 找不到 RGB 模型文件 {RGB_MODEL_PATH}")
        exit()
    # RGB 模型架构必须与 rgb_model.py 一致: r2plus1d_18
    rgb_model = models.r2plus1d_18(weights=None) # 推理时不需要预训练权重
    rgb_model.fc = nn.Linear(rgb_model.fc.in_features, num_classes)
    rgb_model.load_state_dict(torch.load(RGB_MODEL_PATH, map_location=device))
    rgb_model.to(device)
    rgb_model.eval()

    # 4. 获取测试文件列表
    # 优先找 .avi，找不到再找 .mp4
    test_files = glob.glob(os.path.join(TEST_VIDEO_DIR, "*.avi"))
    if len(test_files) == 0: 
        test_files = glob.glob(os.path.join(TEST_VIDEO_DIR, "*.mp4"))
    
    print(f"🔥 开始处理 {len(test_files)} 个测试样本...")
    print(f"⚖️ 融合策略: RGB权重 {ALPHA_RGB} + Skeleton权重 {ALPHA_SKELETON}")
    
    results = []
    
    with torch.no_grad():
        for i, video_path in enumerate(test_files):
            file_id = os.path.splitext(os.path.basename(video_path))[0]
            video_name = file_id + ".avi" # 提交格式要求保持 avi 后缀
            
            # 对应的骨架文件路径
            npy_path = os.path.join(TEST_SKELETON_DIR, file_id + ".npy")
            
            # --- Stream 1: Skeleton 推理 ---
            skel_input = process_skeleton(npy_path).to(device)
            skel_logits = skel_model(skel_input)
            skel_probs = torch.softmax(skel_logits, dim=1) # Logits -> 概率分布
            
            # --- Stream 2: RGB 推理 ---
            rgb_input = process_video(video_path).to(device)
            rgb_logits = rgb_model(rgb_input)
            rgb_probs = torch.softmax(rgb_logits, dim=1)   # Logits -> 概率分布
            
            # --- Late Fusion (加权融合) ---
            # 核心公式: Final_Prob = w1 * P_rgb + w2 * P_skel
            final_probs = (ALPHA_RGB * rgb_probs) + (ALPHA_SKELETON * skel_probs)
            
            # 取概率最大的类别作为预测结果
            _, predicted = torch.max(final_probs, 1)
            label_name = int_to_label[predicted.item()]
            
            results.append([video_name, label_name])
            
            if (i+1) % 50 == 0: print(f"  已处理 {i+1}/{len(test_files)}")

    # 5. 保存结果到 CSV
    out_df = pd.DataFrame(results)
    # 根据作业要求，通常不需要 header，index 也不要
    out_df.to_csv(OUTPUT_FILE, index=False, header=False)
    
    print(f"\n🎉 融合推理完成！")
    print(f"📄 提交文件已生成: {os.path.abspath(OUTPUT_FILE)}")
    print("💡 提示: 请在报告中详细描述这个 'Two-Stream Architecture' 以获得 Originality 加分。")