#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from joblib import load
from tqdm import tqdm

# ========= Configuration Section (modify as needed) =========

YOLO_WEIGHTS = "yolo11n-cls.pt"     # must match the weights used during training
XGB_MODEL_PATH = "xgb_video_cls.pkl"
FRAME_STEP = 5                      # frame sampling interval
CSV_ENCODING = "gbk"                # if label_csv contains Chinese, usually GBK is used
VIDEO_EXTS = [".avi", ".mp4", ".mov", ".mkv"]  # supported video extensions

OUTPUT_CSV = "folder_predictions_with_probs.csv"

# =====================================


def load_label_mapping_from_csv(label_csv_path):
    """
    Read class_id and class_name mapping from the training CSV:
      - class_id: 1 ~ K
      - class_name: category name

    Returns:
        id2name dictionary, where key = 0-based class ID
    """
    df = pd.read_csv(label_csv_path, encoding=CSV_ENCODING)

    if "class_id" not in df.columns or "class_name" not in df.columns:
        raise ValueError("label_csv must contain 'class_id' and 'class_name' columns!")

    id2name = {}
    for _, row in df.iterrows():
        cls_id_raw = int(row["class_id"])
        cls_id = cls_id_raw - 1        # convert to 0-based
        cls_name = str(row["class_name"])
        id2name[cls_id] = cls_name

    return id2name


def extract_frame_embeddings_from_video(model, video_path, frame_step=5):
    """
    Sample frames from a video and extract YOLO11 classification embeddings using model.embed().
    Returns:
        frame_embs -> [T, D]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return None

    frame_embs = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            embs = model.embed(frame, verbose=False)
            emb = embs[0].squeeze().detach().cpu().numpy()  # [D]
            frame_embs.append(emb)

        frame_idx += 1

    cap.release()

    if len(frame_embs) == 0:
        print(f"[ERROR] No embeddings extracted. Please check the video and FRAME_STEP: {video_path}")
        return None

    frame_embs = np.stack(frame_embs, axis=0)  # [T, D]
    return frame_embs


def aggregate_video_features(frame_embs):
    """
    Convert [T, D] → [3D] by concatenating:
      - mean
      - max
      - std
    """
    mean_feat = frame_embs.mean(axis=0)
    max_feat = frame_embs.max(axis=0)
    std_feat = frame_embs.std(axis=0)
    video_feat = np.concatenate([mean_feat, max_feat, std_feat], axis=0)
    return video_feat


def predict_single_video(video_path, model_yolo, model_xgb, id2name=None, frame_step=5):
    """
    Perform action prediction for a single video.

    Returns a dictionary:
      {
        "video_path": ...,
        "video_name": ...,
        "pred_id": ...,
        "pred_name": ... (if available),
        "probs": np.ndarray[num_classes]  # XGBoost probability vector
      }
    """
    frame_embs = extract_frame_embeddings_from_video(model_yolo, video_path, frame_step=frame_step)
    if frame_embs is None:
        return None

    video_feat = aggregate_video_features(frame_embs)   # [3D]
    video_feat = video_feat.reshape(1, -1)              # [1, 3D]

    # XGBoost prediction (class ID)
    pred_id = int(model_xgb.predict(video_feat)[0])

    # XGBoost probability vector (confidence)
    # XGBClassifier supports predict_proba even when using multi:softmax or multi:softprob.
    probs = model_xgb.predict_proba(video_feat)[0]      # [num_classes]

    pred_name = None
    if id2name is not None and pred_id in id2name:
        pred_name = id2name[pred_id]

    return {
        "video_path": video_path,
        "video_name": os.path.basename(video_path),
        "pred_id": pred_id,
        "pred_name": pred_name,
        "probs": probs
    }


def collect_videos_in_folder(folder, exts):
    """
    Traverse a folder and collect all videos with the specified extensions.
    """
    videos = []
    for root, _, files in os.walk(folder):
        for f in files:
            lower = f.lower()
            if any(lower.endswith(ext) for ext in exts):
                videos.append(os.path.join(root, f))
    return sorted(videos)


def main():
    parser = argparse.ArgumentParser(
        description="Action prediction for all videos in a folder: YOLO11 embedding + XGBoost (with confidence vectors)"
    )
    parser.add_argument("--folder", type=str, required=True, help="Folder containing input videos")
    parser.add_argument("--label_csv", type=str, default=None,
                        help="Optional: training CSV for id->class_name mapping")
    parser.add_argument("--frame_step", type=int, default=FRAME_STEP,
                        help="Frame sampling interval (default 5; smaller = finer but slower)")
    args = parser.parse_args()

    folder = args.folder
    if not os.path.isdir(folder):
        print(f"[ERROR] Invalid folder: {folder}")
        return

    # 1. Collect videos
    videos = collect_videos_in_folder(folder, VIDEO_EXTS)
    if len(videos) == 0:
        print(f"[ERROR] No videos found in folder (extensions: {VIDEO_EXTS}): {folder}")
        return

    print(f"[INFO] Found {len(videos)} videos in the folder.")

    # 2. Load YOLO11 classification model
    print("[INFO] Loading YOLO11 classification model...")
    model_yolo = YOLO(YOLO_WEIGHTS)

    # 3. Load XGBoost model
    print("[INFO] Loading XGBoost model...")
    if not os.path.exists(XGB_MODEL_PATH):
        print(f"[ERROR] XGBoost model file not found: {XGB_MODEL_PATH}")
        return
    model_xgb = load(XGB_MODEL_PATH)

    # 4. Load category mapping (optional)
    id2name = None
    if args.label_csv is not None:
        if os.path.exists(args.label_csv):
            print("[INFO] Loading class name mapping from label_csv...")
            id2name = load_label_mapping_from_csv(args.label_csv)
        else:
            print(f"[WARN] label_csv file not found: {args.label_csv}. Only class IDs will be output.")

    # 5. Predict each video
    raw_results = []
    for v in tqdm(videos, desc="Predicting videos"):
        r = predict_single_video(
            v,
            model_yolo,
            model_xgb,
            id2name=id2name,
            frame_step=args.frame_step
        )
        if r is not None:
            raw_results.append(r)

    if len(raw_results) == 0:
        print("[ERROR] No videos were successfully predicted.")
        return

    # 6. Print summary
    print("\n========== Prediction Summary ==========")
    for r in raw_results:
        if r["pred_name"] is not None:
            print(f"{r['video_name']} -> id={r['pred_id']}, name={r['pred_name']}")
        else:
            print(f"{r['video_name']} -> id={r['pred_id']}")
    print("========================================\n")

    # 7. Expand probability vector into prob_0, prob_1, ..., and save to CSV
    rows = []
    for r in raw_results:
        base = {
            "video_path": r["video_path"],
            "video_name": r["video_name"],
            "pred_id": r["pred_id"],
            "pred_name": r["pred_name"],
        }
        probs = r["probs"]
        for i, p in enumerate(probs):
            base[f"prob_{i}"] = float(p)
        rows.append(base)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[INFO] Prediction results (with confidence vectors) saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
