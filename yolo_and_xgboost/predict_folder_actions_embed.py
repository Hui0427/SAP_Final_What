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

# ========= 配置区域（按需修改） =========

YOLO_WEIGHTS = "yolo11n-cls.pt"     # 与训练时相同
XGB_MODEL_PATH = "xgb_video_cls.pkl"
FRAME_STEP = 5                      # 抽帧间隔
CSV_ENCODING = "gbk"                # label_csv 如果有中文，一般是 gbk
VIDEO_EXTS = [".avi", ".mp4", ".mov", ".mkv"]  # 支持的视频后缀

OUTPUT_CSV = "folder_predictions_with_probs.csv"

# =====================================


def load_label_mapping_from_csv(label_csv_path):
    """
    从训练时的 CSV 里读取 class_id 和 class_name 映射：
      - class_id: 1 ~ K
      - class_name: 类别名称
    返回：id2name 字典，key 是 0-based 类别 id
    """
    df = pd.read_csv(label_csv_path, encoding=CSV_ENCODING)

    if "class_id" not in df.columns or "class_name" not in df.columns:
        raise ValueError("label_csv 必须包含 'class_id' 和 'class_name' 列！")

    id2name = {}
    for _, row in df.iterrows():
        cls_id_raw = int(row["class_id"])
        cls_id = cls_id_raw - 1        # 转成 0-based
        cls_name = str(row["class_name"])
        id2name[cls_id] = cls_name

    return id2name


def extract_frame_embeddings_from_video(model, video_path, frame_step=5):
    """
    对一个视频抽帧并用 YOLO11 分类模型的 embed() 提取每帧 embedding。
    返回：frame_embs -> [T, D]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频：{video_path}")
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
        print(f"[ERROR] 未提取到任何 embedding，请检查视频和 FRAME_STEP：{video_path}")
        return None

    frame_embs = np.stack(frame_embs, axis=0)  # [T, D]
    return frame_embs


def aggregate_video_features(frame_embs):
    """
    [T, D] -> [3D]  (mean / max / std)
    """
    mean_feat = frame_embs.mean(axis=0)
    max_feat = frame_embs.max(axis=0)
    std_feat = frame_embs.std(axis=0)
    video_feat = np.concatenate([mean_feat, max_feat, std_feat], axis=0)
    return video_feat


def predict_single_video(video_path, model_yolo, model_xgb, id2name=None, frame_step=5):
    """
    对单个视频进行动作预测。
    返回字典：
      {
        "video_path": ...,
        "video_name": ...,
        "pred_id": ...,
        "pred_name": ... (如有),
        "probs": np.ndarray[num_classes]   # XGBoost 的概率向量
      }
    """
    frame_embs = extract_frame_embeddings_from_video(model_yolo, video_path, frame_step=frame_step)
    if frame_embs is None:
        return None

    video_feat = aggregate_video_features(frame_embs)   # [3D]
    video_feat = video_feat.reshape(1, -1)              # [1, 3D]

    # XGBoost 预测类别 id
    pred_id = int(model_xgb.predict(video_feat)[0])

    # XGBoost 预测概率向量（置信度向量）
    # XGBClassifier 无论 objective 写的是 multi:softmax 还是 multi:softprob，
    # 都可以使用 predict_proba 得到每类概率。
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
    遍历文件夹，收集所有指定后缀的视频文件绝对路径。
    """
    videos = []
    for root, _, files in os.walk(folder):
        for f in files:
            lower = f.lower()
            if any(lower.endswith(ext) for ext in exts):
                videos.append(os.path.join(root, f))
    return sorted(videos)


def main():
    parser = argparse.ArgumentParser(description="文件夹内所有视频动作预测：YOLO11 embedding + XGBoost（含置信度向量）")
    parser.add_argument("--folder", type=str, required=True, help="包含待预测视频的文件夹路径")
    parser.add_argument("--label_csv", type=str, default=None,
                        help="可选：训练集的 CSV（用于 id->类别名 映射）")
    parser.add_argument("--frame_step", type=int, default=FRAME_STEP,
                        help="抽帧间隔（默认 5，越小越精细但越慢）")
    args = parser.parse_args()

    folder = args.folder
    if not os.path.isdir(folder):
        print(f"[ERROR] 不是有效文件夹：{folder}")
        return

    # 1. 收集视频列表
    videos = collect_videos_in_folder(folder, VIDEO_EXTS)
    if len(videos) == 0:
        print(f"[ERROR] 文件夹中没有找到任何视频（后缀 {VIDEO_EXTS}）：{folder}")
        return

    print(f"[INFO] 在文件夹中找到 {len(videos)} 个视频。")

    # 2. 加载 YOLO11 分类模型
    print("[INFO] 加载 YOLO11 分类模型...")
    model_yolo = YOLO(YOLO_WEIGHTS)

    # 3. 加载 XGBoost 模型
    print("[INFO] 加载 XGBoost 模型...")
    if not os.path.exists(XGB_MODEL_PATH):
        print(f"[ERROR] 找不到 XGBoost 模型文件：{XGB_MODEL_PATH}")
        return
    model_xgb = load(XGB_MODEL_PATH)

    # 4. 加载类别映射（可选）
    id2name = None
    if args.label_csv is not None:
        if os.path.exists(args.label_csv):
            print("[INFO] 从 label_csv 加载类别名称映射...")
            id2name = load_label_mapping_from_csv(args.label_csv)
        else:
            print(f"[WARN] label_csv 文件不存在：{args.label_csv}，将只输出类别 id。")

    # 5. 对每个视频做预测
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
        print("[ERROR] 没有任何视频预测成功。")
        return

    # 6. 打印简要结果
    print("\n========== 预测结果汇总 ==========")
    for r in raw_results:
        if r["pred_name"] is not None:
            print(f"{r['video_name']} -> id={r['pred_id']}, name={r['pred_name']}")
        else:
            print(f"{r['video_name']} -> id={r['pred_id']}")
    print("=================================\n")

    # 7. 把概率向量展开成 prob_0, prob_1, ... 列，并保存 CSV
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
    print(f"[INFO] 所有视频预测结果（含置信度向量）已保存到：{OUTPUT_CSV}")


if __name__ == "__main__":
    main()
