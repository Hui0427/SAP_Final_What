#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm

# ========= 配置区域：按需修改 =========

# YOLO11 分类权重
YOLO_WEIGHTS = "yolo11n-cls.pt"

# 视频根目录（所有 .avi 存在的地方）
VIDEO_ROOT = r"E:\A_KCL\Sensing and perception\dataset\train_set"

# 标签 CSV 路径（三列：video_name, class_name, class_id）
CSV_PATH =  r"E:\A_KCL\Sensing and perception\dataset\annotations\train_set_labels.csv"

# 每隔多少帧取一帧
FRAME_STEP = 5

# 视频后缀
VIDEO_EXT = ".avi"

# 输出文件
OUT_FEATURES_PATH = "video_features.npy"
OUT_LABELS_PATH = "video_labels.npy"
OUT_VIDEO_NAMES_PATH = "video_names.npy"

# 如果 CSV 有中文，一般是 gbk；若出错可以改成 "utf-8"
CSV_ENCODING = "gbk"

# ====================================


def load_video_paths_and_labels(csv_path, video_root):
    """
    读取 CSV，返回：
      - video_paths: 视频绝对路径列表
      - labels:      0-based 类别 id 列表（np.int64）
      - video_names: 不带后缀的视频名列表
    """
    df = pd.read_csv(csv_path, encoding=CSV_ENCODING)

    for col in ["video_name", "class_name", "class_id"]:
        if col not in df.columns:
            raise ValueError(f"CSV 必须包含列 'video_name', 'class_name', 'class_id'，"
                             f"当前列名为：{df.columns.tolist()}")

    video_paths = []
    labels = []
    video_names = []

    for _, row in df.iterrows():
        v_name = str(row["video_name"])      # 不带后缀的视频名
        cls_name = str(row["class_name"])    # 仅用于打印
        cls_id_raw = int(row["class_id"])    # 1 ~ K
        cls_id = cls_id_raw - 1              # 转成 0 ~ K-1

        if cls_id < 0:
            raise ValueError(f"发现 class_id_raw={cls_id_raw} < 1，检查 CSV 标注是否正确！")

        filename = v_name + VIDEO_EXT
        v_path = os.path.join(video_root, filename)

        if not os.path.exists(v_path):
            print(f"[WARN] 视频不存在：{v_path} (class_name={cls_name}, class_id_raw={cls_id_raw})")
            continue

        video_paths.append(v_path)
        labels.append(cls_id)
        video_names.append(v_name)

    labels = np.array(labels, dtype=np.int64)
    video_names = np.array(video_names)

    print(f"[INFO] 读取到 {len(video_paths)} 个有效视频。")
    print(f"[INFO] 类别 id 范围（0-based）：min={labels.min()}, max={labels.max()}")

    return video_paths, labels, video_names


def extract_frame_embeddings_from_video(model, video_path, frame_step=5):
    """
    对一个视频抽帧，并用 YOLO11 分类模型的 embed() 提取每帧的 embedding。
    返回：frame_embs -> [T, D]（T 帧数, D embedding 维度）
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] 无法打开视频：{video_path}")
        return None

    frame_embs = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            # Ultralytics YOLO11 提供 model.embed() 直接返回特征向量
            # 文档示例：embedding_vector = model.embed("image.jpg")
            embs = model.embed(frame, verbose=False)   # list-like，长度=1
            emb = embs[0]                              # Tensor shape: [D] 或 [1, D]
            emb = emb.squeeze()                        # 去掉多余维度
            emb = emb.detach().cpu().numpy()           # 转成 numpy [D]
            frame_embs.append(emb)

        frame_idx += 1

    cap.release()

    if len(frame_embs) == 0:
        print(f"[WARN] 视频没有成功提取任何 embedding：{video_path}")
        return None

    frame_embs = np.stack(frame_embs, axis=0)  # [T, D]
    return frame_embs


def aggregate_video_features(frame_embs):
    """
    把 [T, D] 的帧 embedding 聚合成视频级特征：
      - mean / max / std 拼在一起 -> [3D]
    """
    mean_feat = frame_embs.mean(axis=0)   # [D]
    max_feat = frame_embs.max(axis=0)     # [D]
    std_feat = frame_embs.std(axis=0)     # [D]

    video_feat = np.concatenate([mean_feat, max_feat, std_feat], axis=0)  # [3D]
    return video_feat


def main():
    # 1. 加载 YOLO11 分类模型
    print("[INFO] 加载 YOLO11 分类模型...")
    model = YOLO(YOLO_WEIGHTS)

    # 2. 读取视频路径和标签
    print("[INFO] 读取 CSV，加载视频列表和标签...")
    video_paths, labels, video_names = load_video_paths_and_labels(CSV_PATH, VIDEO_ROOT)

    all_feats = []
    valid_labels = []
    valid_names = []

    # 3. 遍历每个视频，抽帧 + 提取 embedding + 聚合
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
        print("[ERROR] 没有成功提取到任何视频特征，请检查路径/CSV/FRAME_STEP。")
        return

    all_feats = np.stack(all_feats, axis=0)           # [N, 3D]
    valid_labels = np.array(valid_labels, dtype=np.int64)
    valid_names = np.array(valid_names)

    # 4. 保存
    np.save(OUT_FEATURES_PATH, all_feats)
    np.save(OUT_LABELS_PATH, valid_labels)
    np.save(OUT_VIDEO_NAMES_PATH, valid_names)

    print(f"[INFO] 特征保存到：{OUT_FEATURES_PATH}，shape={all_feats.shape}")
    print(f"[INFO] 标签保存到：{OUT_LABELS_PATH}，shape={valid_labels.shape}")
    print(f"[INFO] 视频名保存到：{OUT_VIDEO_NAMES_PATH}，shape={valid_names.shape}")
    print("[INFO] 完成。")


if __name__ == "__main__":
    main()
