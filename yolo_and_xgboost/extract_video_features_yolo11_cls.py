import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm

# ======= 配置区域：按你实际情况修改 =======

# YOLO11 分类权重路径
YOLO_WEIGHTS = "yolo11n-cls.pt"

# 存放视频的根目录（所有 .avi 都在这里）
VIDEO_ROOT = r"E:\A_KCL\Sensing and perception\dataset\train_set"

# CSV 路径：包含 video_name、class_name、class_id 三列
CSV_PATH = r"E:\A_KCL\Sensing and perception\dataset\annotations\train_set_labels.csv"

# 每隔多少帧取一帧
FRAME_STEP = 5

# 输出特征文件
OUT_FEATURES_PATH = "video_features.npy"
OUT_LABELS_PATH = "video_labels.npy"
OUT_VIDEO_NAMES_PATH = "video_names.npy"

# 视频扩展名（你说的是 .avi，这里单独写成一个变量，方便之后改）
VIDEO_EXT = ".avi"

# =========================================


def load_video_paths_and_labels(csv_path, video_root):
    """
    读取 CSV，返回：
      - 视频绝对路径列表 video_paths
      - 对应的类别 id 列表 labels（注意这里已经从 1-based 转成 0-based）
      - 视频名（不带后缀）列表 video_names

    要求 CSV 至少包含 3 列：
      - video_name: 不带后缀的视频名，例如 'person01_run_001'
      - class_name: 类别名称（这里不参与训练，只是方便你查看）
      - class_id:   类别 id，从 1 开始（1 ~ num_classes），这里会减 1 变成 0 ~ num_classes-1
    """
    # 如果你的 CSV 是 GBK 编码，可以把 encoding 改成 "gbk"
    df = pd.read_csv(csv_path, encoding="utf-8")

    for col in ["video_name", "class_name", "class_id"]:
        if col not in df.columns:
            raise ValueError(f"CSV 必须包含列 '{col}'，当前列名为：{df.columns.tolist()}")

    video_paths = []
    labels = []
    video_names = []

    for _, row in df.iterrows():
        v_name = str(row["video_name"])          # 不带后缀的视频名
        cls_name = str(row["class_name"])        # 目前只是读取，不强制使用

        # ---- 关键：class_id 从 1 开始，这里减 1 变成 0-based ----
        cls_id_raw = int(row["class_id"])        # 1, 2, 3, ..., K
        cls_id = cls_id_raw - 1                  # 0, 1, 2, ..., K-1

        if cls_id < 0:
            raise ValueError(f"发现 class_id_raw={cls_id_raw} < 1，检查 CSV 标注是否正确！")

        # 拼接成完整文件名：video_name + ".avi"
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

    # 打印一下类别范围，确认一下
    print(f"[INFO] 读取到的类别 id 范围（0-based）：min={labels.min()}, max={labels.max()}")

    return video_paths, labels, video_names



def extract_frame_probs_from_video(model, video_path, frame_step=5):
    """
    对一个视频抽帧并用 YOLO11 分类推理。
    返回：frame_probs -> shape [T, C]（T 是帧数，C 是类别数）
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] 无法打开视频：{video_path}")
        return None

    frame_probs = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            # YOLO 模型推理
            results = model(frame, verbose=False)
            # 分类任务：results[0].probs.data 为 softmax 概率
            probs = results[0].probs.data.cpu().numpy()  # [C]
            frame_probs.append(probs)

        frame_idx += 1

    cap.release()

    if len(frame_probs) == 0:
        return None

    frame_probs = np.stack(frame_probs, axis=0)  # [T, C]
    return frame_probs


def aggregate_video_features(frame_probs):
    """
    把 [T, C] 的帧概率矩阵汇聚成一个视频级特征向量。
    使用 mean / max / std 三种统计，然后拼接：[3C]
    """
    mean_feat = frame_probs.mean(axis=0)   # [C]
    max_feat = frame_probs.max(axis=0)     # [C]
    std_feat = frame_probs.std(axis=0)     # [C]

    video_feat = np.concatenate([mean_feat, max_feat, std_feat], axis=0)  # [3C]
    return video_feat


def main():
    # 1. 加载 YOLO11 分类模型
    print("[INFO] 加载 YOLO11 分类模型...")
    model = YOLO(YOLO_WEIGHTS)

    # 2. 读取视频路径和标签
    print("[INFO] 读取 CSV，加载视频列表和标签...")
    video_paths, labels, video_names = load_video_paths_and_labels(CSV_PATH, VIDEO_ROOT)
    print(f"[INFO] 有效视频数量：{len(video_paths)}")

    all_feats = []
    valid_labels = []
    valid_names = []

    # 3. 遍历每个视频，抽帧 + 概率统计
    for v_path, label, name in tqdm(
        list(zip(video_paths, labels, video_names)),
        total=len(video_paths),
        desc="Extracting video features"
    ):
        frame_probs = extract_frame_probs_from_video(model, v_path, frame_step=FRAME_STEP)

        if frame_probs is None:
            print(f"[WARN] 视频帧特征为空，跳过：{v_path}")
            continue

        video_feat = aggregate_video_features(frame_probs)  # [3C]
        all_feats.append(video_feat)
        valid_labels.append(label)
        valid_names.append(name)

    if len(all_feats) == 0:
        print("[ERROR] 没有成功提取到任何视频特征，请检查视频路径和 CSV 设置。")
        return

    all_feats = np.stack(all_feats, axis=0)           # [N, 3C]
    valid_labels = np.array(valid_labels, dtype=np.int64)
    valid_names = np.array(valid_names)

    # 4. 保存到文件
    np.save(OUT_FEATURES_PATH, all_feats)
    np.save(OUT_LABELS_PATH, valid_labels)
    np.save(OUT_VIDEO_NAMES_PATH, valid_names)

    print(f"[INFO] 特征保存到：{OUT_FEATURES_PATH}，shape={all_feats.shape}")
    print(f"[INFO] 标签保存到：{OUT_LABELS_PATH}，shape={valid_labels.shape}")
    print(f"[INFO] 视频名保存到：{OUT_VIDEO_NAMES_PATH}，shape={valid_names.shape}")
    print("[INFO] 完成。")


if __name__ == "__main__":
    main()
