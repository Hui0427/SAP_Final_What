#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm

# ========= Configuration Section: modify as needed =========

# YOLO11 classification weights
YOLO_WEIGHTS = "yolo11n-cls.pt"

# Root directory containing videos (all .avi files are stored here)
VIDEO_ROOT = r"E:\A_KCL\Sensing and perception\dataset\train_set"

# Label CSV path (three columns: video_name, class_name, class_id)
CSV_PATH = r"E:\A_KCL\Sensing and perception\dataset\annotations\train_set_labels.csv"

# Sample one frame every N frames
FRAME_STEP = 5

# Video extension
VIDEO_EXT = ".avi"

# Output files
OUT_FEATURES_PATH = "video_features.npy"
OUT_LABELS_PATH = "video_labels.npy"
OUT_VIDEO_NAMES_PATH = "video_names.npy"

# If the CSV contains Chinese characters, it may use GBK encoding;
# change to "utf-8" if needed.
CSV_ENCODING = "gbk"

# ====================================


def load_video_paths_and_labels(csv_path, video_root):
    """
    Read the CSV and return:
      - video_paths: list of absolute video paths
      - labels:      list of 0-based class IDs (np.int64)
      - video_names: list of video names without extension
    """
    df = pd.read_csv(csv_path, encoding=CSV_ENCODING)

    for col in ["video_name", "class_name", "class_id"]:
        if col not in df.columns:
            raise ValueError(
                f"CSV must contain columns 'video_name', 'class_name', 'class_id'. "
                f"Current columns: {df.columns.tolist()}"
            )

    video_paths = []
    labels = []
    video_names = []

    for _, row in df.iterrows():
        v_name = str(row["video_name"])      # video name without extension
        cls_name = str(row["class_name"])    # used only for logging
        cls_id_raw = int(row["class_id"])    # 1 ~ K
        cls_id = cls_id_raw - 1              # convert to 0 ~ K-1

        if cls_id < 0:
            raise ValueError(
                f"Detected class_id_raw={cls_id_raw} < 1. Please check your CSV annotations!"
            )

        filename = v_name + VIDEO_EXT
        v_path = os.path.join(video_root, filename)

        if not os.path.exists(v_path):
            print(f"[WARN] Video not found: {v_path} (class_name={cls_name}, class_id_raw={cls_id_raw})")
            continue

        video_paths.append(v_path)
        labels.append(cls_id)
        video_names.append(v_name)

    labels = np.array(labels, dtype=np.int64)
    video_names = np.array(video_names)

    print(f"[INFO] Loaded {len(video_paths)} valid videos.")
    print(f"[INFO] Class ID range (0-based): min={labels.min()}, max={labels.max()}")

    return video_paths, labels, video_names


def extract_frame_embeddings_from_video(model, video_path, frame_step=5):
    """
    Sample frames from a video and extract YOLO11 classification embeddings.
    Returns: frame_embs -> [T, D] (T frames, D embedding dimension)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return None

    frame_embs = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            # Ultralytics YOLO11 provides model.embed() which returns the feature vector
            # Example in docs: embedding_vector = model.embed("image.jpg")
            embs = model.embed(frame, verbose=False)   # list-like, length=1
            emb = embs[0]                              # Tensor shape: [D] or [1, D]
            emb = emb.squeeze()                        # remove extra dimension if any
            emb = emb.detach().cpu().numpy()           # convert to numpy [D]
            frame_embs.append(emb)

        frame_idx += 1

    cap.release()

    if len(frame_embs) == 0:
        print(f"[WARN] No embeddings extracted from video: {video_path}")
        return None

    frame_embs = np.stack(frame_embs, axis=0)  # [T, D]
    return frame_embs


def aggregate_video_features(frame_embs):
    """
    Aggregate [T, D] frame embeddings into a single video-level feature:
      - concatenate mean / max / std  -> [3D]
    """
    mean_feat = frame_embs.mean(axis=0)   # [D]
    max_feat = frame_embs.max(axis=0)     # [D]
    std_feat = frame_embs.std(axis=0)     # [D]

    video_feat = np.concatenate([mean_feat, max_feat, std_feat], axis=0)  # [3D]
    return video_feat


def main():
    # 1. Load YOLO11 classification model
    print("[INFO] Loading YOLO11 classification model...")
    model = YOLO(YOLO_WEIGHTS)

    # 2. Load video paths and labels from CSV
    print("[INFO] Reading CSV and loading video list + labels...")
    video_paths, labels, video_names = load_video_paths_and_labels(CSV_PATH, VIDEO_ROOT)

    all_feats = []
    valid_labels = []
    valid_names = []

    # 3. Iterate through each video: sampling + embed extraction + aggregation
    for v_path, label, name in tqdm(
        list(zip(video_paths, labels, video_names)),
        total=len(video_paths),
        desc="Extracting video embeddings"
    ):
        frame_embs = extract_frame_embeddings_from_video(model, v_path, frame_step=FRAME_STEP)
        if frame_embs is None:
            continue

        video_feat = aggregate_video_features(frame_embs)  # [3D]
        all_feats.append(video_feat)
        valid_labels.append(label)
        valid_names.append(name)

    if len(all_feats) == 0:
        print("[ERROR] No video features extracted. Please check paths/CSV/FRAME_STEP.")
        return

    all_feats = np.stack(all_feats, axis=0)           # [N, 3D]
    valid_labels = np.array(valid_labels, dtype=np.int64)
    valid_names = np.array(valid_names)

    # 4. Save results
    np.save(OUT_FEATURES_PATH, all_feats)
    np.save(OUT_LABELS_PATH, valid_labels)
    np.save(OUT_VIDEO_NAMES_PATH, valid_names)

    print(f"[INFO] Features saved to: {OUT_FEATURES_PATH}, shape={all_feats.shape}")
    print(f"[INFO] Labels saved to: {OUT_LABELS_PATH}, shape={valid_labels.shape}")
    print(f"[INFO] Video names saved to: {OUT_VIDEO_NAMES_PATH}, shape={valid_names.shape}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
