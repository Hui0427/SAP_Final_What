import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from typing import List, Optional, Dict, Any


# ========================= 配置区 =========================

# 训练好的分类模型路径（best.pt）
MODEL_PATH = Path(
    r"E:\A_KCL\Sensing and perception\A_GroupCoursework\yolo\runs\classify\train4\weights\best.pt"
)

# ---------- 有标签的 train 视频，用来算准确率 ----------
# train 视频所在根目录（原始视频，不是抽帧图片）
TRAIN_VIDEO_ROOT = Path(
    r"E:\A_KCL\Sensing and perception\dataset\train_set"
)

# train 标签 Excel（和你之前抽帧用的是同一个）
TRAIN_LABELS_EXCEL = Path(
    r"E:\A_KCL\Sensing and perception\dataset\annotations\train_set_labels.csv"
)

# Excel 中的列名（根据你自己的表头修改）
COL_VIDEO_NAME = "video_name"      # 不带后缀的文件名，例如 "xxx_0001"
COL_CLASS_NAME = "class_name_en"   # 类别英文名，例如 "walking"

# 评估 train 集时，结果保存的 CSV
TRAIN_EVAL_CSV = Path(
    r"E:\A_KCL\Sensing and perception\A_GroupCoursework\yolo\train_eval_vote_compare.csv"
)

# ---------- 无标签的 test 视频，只做预测 ----------
VIDEO_ROOT = Path(
    r"E:\A_KCL\Sensing and perception\dataset\test_set"
)

# test 预测结果 CSV
OUTPUT_CSV = Path(
    r"E:\A_KCL\Sensing and perception\A_GroupCoursework\yolo\video_predictions_vote_compare.csv"
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
) -> Optional[Dict[str, Any]]:
    """
    对单个视频做动作预测：
    - 按 frame_step 抽帧
    - 对每帧做分类
    - 使用两种策略得到视频级别预测：
      1) 平均概率投票（prob-vote）
      2) 多数投票（majority-vote）

    返回一个字典，包含两种策略的预测结果。
    如果视频无法读取或没有有效帧，返回 None。
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

    # 批量预测
    results = model.predict(
        frames,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )

    probs_list = []
    frame_pred_ids = []
    frame_pred_scores = []

    for r in results:
        if r.probs is None:
            continue
        probs = r.probs.data.cpu().numpy()  # [num_classes]
        probs_list.append(probs)

        # 每帧的 top1 预测
        cls_id_frame = int(np.argmax(probs))
        score_frame = float(probs[cls_id_frame])
        frame_pred_ids.append(cls_id_frame)
        frame_pred_scores.append(score_frame)

    if not probs_list:
        print(f"[INFO] 视频未获得任何概率输出: {video_path}")
        return None

    probs_arr = np.stack(probs_list, axis=0)   # [T, num_classes]
    num_classes = probs_arr.shape[1]

    # ---------- 策略 1：平均概率投票 ----------
    mean_probs = probs_arr.mean(axis=0)        # [num_classes]
    prob_cls_id = int(np.argmax(mean_probs))
    prob_cls_name = model.names[prob_cls_id]
    prob_score = float(mean_probs[prob_cls_id])

    # ---------- 策略 2：多数投票 ----------
    counts = np.bincount(frame_pred_ids, minlength=num_classes)
    maj_cls_id = int(np.argmax(counts))
    maj_cls_name = model.names[maj_cls_id]
    maj_support = counts[maj_cls_id] / len(frame_pred_ids)  # 多数类占所有帧的比例

    maj_scores = [s for cid, s in zip(frame_pred_ids, frame_pred_scores) if cid == maj_cls_id]
    maj_avg_score = float(np.mean(maj_scores)) if maj_scores else 0.0

    agree = (prob_cls_id == maj_cls_id)

    return {
        "prob_cls_id": prob_cls_id,
        "prob_cls_name": prob_cls_name,
        "prob_score": prob_score,
        "maj_cls_id": maj_cls_id,
        "maj_cls_name": maj_cls_name,
        "maj_support": maj_support,
        "maj_avg_score": maj_avg_score,
        "agree": agree,
        "frame_count": len(frames),
    }


def evaluate_train_set(
    model: YOLO,
    train_video_root: Path,
    labels_excel: Path,
    out_csv: Path,
) -> None:
    """
    在有标签的 train 集上评估：
    - 从 Excel 读取每个视频的真实类别
    - 对每个视频做预测（平均概率 & 多数投票）
    - 计算两种策略在 train 集上的准确率
    - 把详细结果写入 CSV
    """
    if not train_video_root.exists():
        print(f"[WARN] TRAIN_VIDEO_ROOT 不存在，跳过 train 集评估: {train_video_root}")
        return
    if not labels_excel.exists():
        print(f"[WARN] TRAIN_LABELS_EXCEL 不存在，跳过 train 集评估: {labels_excel}")
        return

    print("\n===== 在有标签的 train 视频上评估投票策略 =====")
    print("读取标签 Excel:", labels_excel)

    df = pd.read_excel(labels_excel)
    if COL_VIDEO_NAME not in df.columns or COL_CLASS_NAME not in df.columns:
        raise ValueError(f"Excel 中必须包含列 {COL_VIDEO_NAME!r} 和 {COL_CLASS_NAME!r}")

    # 构建 video_stem -> gt_class_name 映射
    gt_map = {}
    for _, row in df.iterrows():
        stem = str(row[COL_VIDEO_NAME]).strip()
        cls_name = str(row[COL_CLASS_NAME]).strip()
        if stem:
            gt_map[stem] = cls_name

    print(f"从 Excel 读取到 {len(gt_map)} 条标签。")
    video_files = list_videos(train_video_root)
    print(f"在 {train_video_root} 下找到 {len(video_files)} 个视频。")

    records = []
    total = 0
    correct_prob = 0
    correct_maj = 0

    for i, vp in enumerate(video_files, 1):
        stem = vp.stem  # 不带后缀的文件名
        if stem not in gt_map:
            print(f"[INFO] 视频 {vp.name} 在 Excel 中没有标签，跳过。")
            continue

        gt_cls_name = gt_map[stem]
        total += 1
        print(f"[train {total}] 视频: {vp.name}, GT: {gt_cls_name}")

        result = predict_single_video(model, vp)
        if result is None:
            continue

        prob_correct = (result["prob_cls_name"] == gt_cls_name)
        maj_correct = (result["maj_cls_name"] == gt_cls_name)

        if prob_correct:
            correct_prob += 1
        if maj_correct:
            correct_maj += 1

        print(
            f"    平均概率: {result['prob_cls_name']} (score={result['prob_score']:.4f}), "
            f"{'✔' if prob_correct else '✘'}"
        )
        print(
            f"    多数投票: {result['maj_cls_name']} "
            f"(support={result['maj_support']:.3f}, avg_score={result['maj_avg_score']:.4f}), "
            f"{'✔' if maj_correct else '✘'}"
        )

        records.append(
            {
                "video_path": str(vp),
                "video_name": vp.name,
                "video_stem": stem,
                "frame_count": result["frame_count"],
                "gt_class_name": gt_cls_name,
                # 平均概率投票
                "prob_class_name": result["prob_cls_name"],
                "prob_score": result["prob_score"],
                "prob_correct": prob_correct,
                # 多数投票
                "maj_class_name": result["maj_cls_name"],
                "maj_support": result["maj_support"],
                "maj_avg_score": result["maj_avg_score"],
                "maj_correct": maj_correct,
                # 两者是否预测同一类
                "agree_prob_maj": result["agree"],
            }
        )

    if total == 0:
        print("[WARN] 没有任何带标签的视频被评估，请检查路径和 Excel。")
        return

    prob_acc = correct_prob / total
    maj_acc = correct_maj / total

    print("\n===== train 集评估结果 =====")
    print(f"总视频数: {total}")
    print(f"平均概率投票  准确率: {prob_acc:.4f}  ({correct_prob}/{total})")
    print(f"多数投票      准确率: {maj_acc:.4f}  ({correct_maj}/{total})")

    df_out = pd.DataFrame(records)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"train 评估明细已保存到: {out_csv.resolve()}")


def main():
    print("加载模型中:", MODEL_PATH)
    model = YOLO(str(MODEL_PATH))

    # 1) 先在有标签的 train 集上评估两种投票策略
    evaluate_train_set(
        model=model,
        train_video_root=TRAIN_VIDEO_ROOT,
        labels_excel=TRAIN_LABELS_EXCEL,
        out_csv=TRAIN_EVAL_CSV,
    )

    # 2) 再对无标签的 test 集做预测（保持原有功能）
    print("\n===== 对无标签的 test 视频做预测 =====")
    print("搜索视频中:", VIDEO_ROOT)
    video_files = list_videos(VIDEO_ROOT)
    if not video_files:
        print("[ERROR] 未在 VIDEO_ROOT 下找到任何视频，请检查路径和后缀。")
        return

    print(f"共找到 {len(video_files)} 个视频。开始预测...\n")

    records = []
    for i, vp in enumerate(video_files, 1):
        print(f"[test {i}/{len(video_files)}] 处理视频: {vp}")
        result = predict_single_video(model, vp)
        if result is None:
            continue

        print(
            f"    平均概率投票:  {result['prob_cls_name']} "
            f"(id={result['prob_cls_id']}, score={result['prob_score']:.4f})"
        )
        print(
            f"    多数投票结果:  {result['maj_cls_name']} "
            f"(id={result['maj_cls_id']}, support={result['maj_support']:.3f}, "
            f"avg_score={result['maj_avg_score']:.4f})"
        )
        print(f"    两者是否一致: {result['agree']}")

        records.append(
            {
                "video_path": str(vp),
                "video_name": vp.name,
                "frame_count": result["frame_count"],
                # 平均概率投票
                "prob_class_id": result["prob_cls_id"],
                "prob_class_name": result["prob_cls_name"],
                "prob_score": result["prob_score"],
                # 多数投票
                "maj_class_id": result["maj_cls_id"],
                "maj_class_name": result["maj_cls_name"],
                "maj_support": result["maj_support"],
                "maj_avg_score": result["maj_avg_score"],
                # 对比
                "agree": result["agree"],
            }
        )

    if not records:
        print("[WARN] 没有任何 test 视频成功预测，可能全部出错，请检查日志。")
        return

    df = pd.DataFrame(records)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n全部 test 视频预测完成！结果已保存到：")
    print(OUTPUT_CSV.resolve())


if __name__ == "__main__":
    main()
