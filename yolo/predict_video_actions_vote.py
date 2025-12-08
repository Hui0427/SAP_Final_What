import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from typing import List, Tuple, Optional


# ========================= 配置区 =========================

# 训练好的分类模型路径（best.pt）
MODEL_PATH = Path(
    r"E:\A_KCL\Sensing and perception\A_GroupCoursework\yolo\runs\classify\train4\weights\best.pt"
)

# 要做预测的视频根目录（会递归遍历下面所有视频）
VIDEO_ROOT = Path(
    r"E:\A_KCL\Sensing and perception\dataset\test_set"
)

# 结果保存的 CSV 路径
OUTPUT_CSV = Path(
    r"E:\A_KCL\Sensing and perception\A_GroupCoursework\yolo\video_predictions.csv"
)

# 每隔多少帧取一帧做分类（2 秒视频建议 1~3 都可以）
FRAME_STEP = 2

# YOLO 分类模型的输入尺寸（一般 224 即可）
IMG_SIZE = 224

# 使用设备：0 = 第一块 GPU，"cpu" = 只用 CPU
DEVICE = 0  # 如果 GPU 有问题可以改成 "cpu"

# 支持的视频后缀
VIDEO_EXTS = [".avi", ".mp4", ".mov", ".mkv"]

# ========================================================


def list_videos(root: Path) -> List[Path]:
    """递归列出根目录下所有视频文件"""
    videos = []
    for ext in VIDEO_EXTS:
        videos.extend(root.rglob(f"*{ext}"))
    return sorted(videos)


def predict_single_video(
    model: YOLO,
    video_path: Path,
    frame_step: int = FRAME_STEP,
    imgsz: int = IMG_SIZE,
    device=DEVICE,
) -> Optional[Tuple[int, str, float, np.ndarray]]:
    """
    对单个视频做动作预测：
    - 按 frame_step 抽帧
    - 对每帧做分类
    - 对所有帧的概率取平均，得到视频级别预测

    返回：
        (class_id, class_name, score, mean_probs)
        如果视频无法读取，返回 None
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] 无法打开视频: {video_path}")
        return None

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            frames.append(frame)  # BGR 格式，YOLO 可以直接吃
        frame_idx += 1

    cap.release()

    if not frames:
        print(f"[INFO] 视频没有有效帧: {video_path}")
        return None

    # 批量预测，避免一帧一帧太慢
    results = model.predict(
        frames,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )

    probs_list = []
    for r in results:
        if r.probs is None:
            continue
        probs = r.probs.data.cpu().numpy()
        probs_list.append(probs)

    if not probs_list:
        print(f"[INFO] 视频未获得任何概率输出: {video_path}")
        return None

    probs_arr = np.stack(probs_list, axis=0)       # [T, num_classes]
    mean_probs = probs_arr.mean(axis=0)            # [num_classes]

    cls_id = int(np.argmax(mean_probs))
    cls_name = model.names[cls_id]
    score = float(mean_probs[cls_id])

    return cls_id, cls_name, score, mean_probs


def main():
    print("加载模型中:", MODEL_PATH)
    model = YOLO(str(MODEL_PATH))

    print("搜索视频中:", VIDEO_ROOT)
    video_files = list_videos(VIDEO_ROOT)
    if not video_files:
        print("[ERROR] 未在 VIDEO_ROOT 下找到任何视频，请检查路径和后缀。")
        return

    print(f"共找到 {len(video_files)} 个视频。开始预测...\n")

    records = []

    for i, vp in enumerate(video_files, 1):
        print(f"[{i}/{len(video_files)}] 处理视频: {vp}")
        result = predict_single_video(model, vp)
        if result is None:
            continue

        cls_id, cls_name, score, mean_probs = result

        print(f"    预测结果: class_id={cls_id}, class_name={cls_name}, score={score:.4f}")

        records.append(
            {
                "video_path": str(vp),
                "video_name": vp.name,
                "class_id": cls_id,
                "class_name": cls_name,
                "score": score,
                # 也可以把整个概率向量保存成字符串（可选）
                "probs": ";".join([f"{p:.6f}" for p in mean_probs]),
            }
        )

    if not records:
        print("[WARN] 没有任何视频成功预测，可能全部出错，请检查日志。")
        return

    df = pd.DataFrame(records)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n全部视频预测完成！结果已保存到：")
    print(OUTPUT_CSV.resolve())


if __name__ == "__main__":
    main()
