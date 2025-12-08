import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from skeleton_lstm_model import PoseLSTM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 64
MODEL_CKPT = "checkpoints/pose_lstm_best.pt"
YOLO_MODEL_PATH = "yolo11n-pose.pt"

def extract_keypoints_for_video(video_path, yolo_model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return None

    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        results = yolo_model(frame, verbose=False)[0]
        if results.keypoints is None or len(results.keypoints) == 0:
            if len(all_frames) > 0:
                all_frames.append(np.zeros_like(all_frames[-1]))
            else:
                all_frames.append(np.zeros(17 * 2, dtype=np.float32))
            continue

        boxes = results.boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        main_id = int(np.argmax(areas))

        kpts = results.keypoints[main_id].xy.cpu().numpy()  # [J,2]
        kpts[:, 0] /= w
        kpts[:, 1] /= h
        frame_vec = kpts.reshape(-1).astype(np.float32)
        all_frames.append(frame_vec)

    cap.release()
    if len(all_frames) == 0:
        return None
    seq = np.stack(all_frames, axis=0)
    return seq  # [T, D]

def resample_seq(seq, seq_len=64):
    T, D = seq.shape
    if T == seq_len:
        return seq
    if T > seq_len:
        idxs = np.linspace(0, T - 1, seq_len).astype(int)
        return seq[idxs]
    else:
        out = np.zeros((seq_len, D), dtype=np.float32)
        out[:T] = seq
        return out

def main(video_path):
    # 加载模型
    ckpt = torch.load(MODEL_CKPT, map_location=DEVICE)
    input_dim = ckpt["input_dim"]
    num_classes = ckpt["num_classes"]
    seq_len = ckpt["seq_len"]

    model = PoseLSTM(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=2,
        num_classes=num_classes,
        bidirectional=True,
        dropout=0.3,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    yolo_model = YOLO(YOLO_MODEL_PATH)

    # 抽关键点
    seq = extract_keypoints_for_video(video_path, yolo_model)
    if seq is None:
        print("No keypoints for this video.")
        return

    seq = resample_seq(seq, seq_len=seq_len)
    seq_tensor = torch.from_numpy(seq).unsqueeze(0).to(DEVICE)  # [1, L, D]

    with torch.no_grad():
        logits = model(seq_tensor)          # [1, C]
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_class = int(np.argmax(probs))
        pred_prob = float(probs[pred_class])

    print(f"Pred class id: {pred_class}, prob: {pred_prob:.4f}")
    # 如果你有 label_map.csv，也可以反查 label_name

if __name__ == "__main__":
    test_video = "dataset/videos/v_test.mp4"  # 换成你的路径
    main(test_video)
