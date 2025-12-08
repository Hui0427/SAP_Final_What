# extract_keypoints_yolo11_pose.py
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO  # yolo11-pose 用 ultralytics 风格

# ===== 根据你自己的情况改这几个路径 =====
VIDEO_ROOT = r"E:\A_KCL\Sensing and perception\dataset"             # 根目录
VIDEO_DIR  = r"E:\A_KCL\Sensing and perception\dataset\train_set"      # 存放视频文件的文件夹
CSV_PATH   = r"E:\A_KCL\Sensing and perception\dataset\annotations\train_set_labels.csv"   # 你的 train.csv
OUT_DIR    = "keypoints"           # 保存关键点 .npy 的目录
MODEL_PATH = "yolo11n-pose.pt"     # 你的 yolo11-pose 权重
# =======================================

os.makedirs(OUT_DIR, exist_ok=True)


def load_labels(csv_path, video_dir):
    """
    读取 train.csv
    期望列：video_name, class_name_en, class_id
    输出的 df 会额外多一列 video_path（完整路径）
    """
    df = None
    last_err = None

    # 依次尝试几种常见编码
    for enc in ["utf-8", "utf-8-sig", "gbk", "latin1"]:
        try:
            print(f"尝试用编码 {enc} 读取 CSV ...")
            # sep=None + engine="python"：自动猜逗号/分号/制表符
            df = pd.read_csv(csv_path, encoding=enc, sep=None, engine="python")
            print(f"用编码 {enc} 读取 CSV 成功")
            break
        except UnicodeDecodeError as e:
            print(f"编码 {enc} 读取失败：{e}")
            last_err = e
            df = None

    if df is None:
        # 所有编码都失败，直接抛出最后一个错误
        raise last_err

    # 确保有这三列
    required_cols = {"video_name", "class_name_en", "class_id"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV 必须包含列: {required_cols}，当前列为: {df.columns.tolist()}")

    # class_id 转成 int
    df["class_id"] = df["class_id"].astype(int)

    # 生成完整视频路径：如果 video_name 里已经带有路径，就直接用
    def make_video_path(name):
        # 如果已经是绝对路径，直接返回
        if os.path.isabs(name):
            return name
        # 如果里面已经有 / 或 \，说明带了子目录，直接接在 VIDEO_ROOT 后面
        if ("/" in name) or ("\\" in name):
            return os.path.join(VIDEO_ROOT, name)
        # 否则就是单纯文件名，拼在 VIDEO_DIR 下
        return os.path.join(video_dir, name)

    df["video_path"] = df["video_name"].apply(make_video_path)

    return df


def extract_video_keypoints(model, video_path, max_frames=None):
    """
    对单个视频抽关键点，返回 numpy 数组： [T, num_joints*2]
    这里只保留 (x, y) 并做 [0,1] 归一化；只取 bbox 最大的那个人。
    """
    cap = cv2.VideoCapture(video_path + ".avi")
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return None

    all_frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if max_frames is not None and frame_idx > max_frames:
            break

        h, w = frame.shape[:2]

        # YOLOv11 pose 推理
        results = model(frame, verbose=False)[0]

        if results.keypoints is None or len(results.keypoints) == 0:
            # 没检测到人，用 0 向量占位
            if len(all_frames) > 0:
                all_frames.append(np.zeros_like(all_frames[-1]))
            else:
                # 第一帧就没检测到，这里假设 17 关键点（不影响后面训练，
                # 后面一旦有真实帧会覆盖这个维度）
                all_frames.append(np.zeros(17 * 2, dtype=np.float32))
            continue

        # 选择 bbox 最大的那个人
        boxes = results.boxes.xyxy.cpu().numpy()  # [N, 4]
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        main_id = int(np.argmax(areas))

        kpts = results.keypoints[main_id].xy.cpu().numpy()  # [num_joints, 2]
        # 归一化到 [0,1]
        kpts[:, 0] /= w
        kpts[:, 1] /= h

        frame_vec = kpts.reshape(-1).astype(np.float32)  # [num_joints*2]
        all_frames.append(frame_vec)

    cap.release()

    if len(all_frames) == 0:
        return None

    seq = np.stack(all_frames, axis=0)  # [T, D]
    return seq


def main():
    # 1. 读取 train.csv
    df = load_labels(CSV_PATH, VIDEO_DIR)
    print("Loaded labels, total videos:", len(df))

    # 2. 加载 YOLOv11-pose 模型
    model = YOLO(MODEL_PATH)

    meta = []  # 保存每个样本的信息：npy_path, label_id, n_frames

    # 3. 遍历每一行：一行 = 一个视频
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        video_path = row["video_path"]
        class_id   = int(row["class_id"])
        video_name = row["video_name"]

        # 输出的 npy 文件名：用 video_name 改后缀
        base_name = os.path.splitext(os.path.basename(video_name))[0]
        out_name  = base_name + ".npy"
        out_path  = os.path.join(OUT_DIR, out_name)

        if os.path.exists(out_path):
            # 已经提取过，直接读取
            seq = np.load(out_path)
            n_frames = seq.shape[0]
        else:
            seq = extract_video_keypoints(model, video_path)
            if seq is None:
                print(f"Skip {video_path}, no keypoints.")
                continue
            np.save(out_path, seq)
            n_frames = seq.shape[0]

        meta.append({
            "npy_path": out_path,
            "label_id": class_id,   # 注意：这里直接用 class_id
            "n_frames": n_frames
        })

    # 4. 保存 meta.csv，后面 LSTM 训练只看这个
    meta_df = pd.DataFrame(meta)
    meta_csv_path = os.path.join(OUT_DIR, "meta.csv")
    meta_df.to_csv(meta_csv_path, index=False)
    print("Saved meta to", meta_csv_path)

    # 5. （可选）保存 class_id ↔ class_name_en 对照表
    label_map_df = df[["class_id", "class_name_en"]].drop_duplicates().sort_values("class_id")
    label_map_path = os.path.join(OUT_DIR, "label_map.csv")
    label_map_df.to_csv(label_map_path, index=False)
    print("Saved label_map.csv to", label_map_path)


if __name__ == "__main__":
    main()
