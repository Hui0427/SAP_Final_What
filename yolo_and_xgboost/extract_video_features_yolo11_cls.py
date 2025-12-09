import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm

# ======= Configuration Section: modify according to your setup =======

# YOLO11 classification weights
YOLO_WEIGHTS = "yolo11n-cls.pt"

# Root directory containing videos (all .avi files are located here)
VIDEO_ROOT = r"E:\A_KCL\Sensing and perception\dataset\train_set"

# CSV path: must include three columns: video_name, class_name, class_id
CSV_PATH = r"E:\A_KCL\Sensing and perception\dataset\annotations\train_set_labels.csv"

# Extract one frame every N frames
FRAME_STEP = 5

# Output feature files
OUT_FEATURES_PATH = "video_features.npy"
OUT_LABELS_PATH = "video_labels.npy"
OUT_VIDEO_NAMES_PATH = "video_names.npy"

# Video extension (you mentioned .avi, so it's extracted as a variable)
VIDEO_EXT = ".avi"

# ================================================================


def load_video_paths_and_labels(csv_path, video_root):
    """
    Read the CSV and return:
      - video_paths: list of absolute video paths
      - labels: class IDs (converted from 1-based to 0-based)
      - video_names: video name without extension

    The CSV must contain at least 3 columns:
      - video_name: name without extension, e.g., 'person01_run_001'
      - class_name: class label (not used for training here, just for reference)
      - class_id:   class ID starting from 1 (1 ~ K). We convert to 0 ~ K-1.
    """
    # If your CSV is in GBK encoding, change encoding="gbk"
    df = pd.read_csv(csv_path, encoding="utf-8")

    for col in ["video_name", "class_name", "class_id"]:
        if col not in df.columns:
            raise ValueError(f"CSV must contain column '{col}', current columns: {df.columns.tolist()}")

    video_paths = []
    labels = []
    video_names = []

    for _, row in df.iterrows():
        v_name = str(row["video_name"])      # video name without extension
        cls_name = str(row["class_name"])    # read for reference

        # ---- class_id starts from 1; convert to 0-based index ----
        cls_id_raw = int(row["class_id"])    # 1, 2, ..., K
        cls_id = cls_id_raw - 1              # 0, 1, ..., K-1

        if cls_id < 0:
            raise ValueError(f"Found class_id_raw={cls_id_raw} < 1. Check your CSV annotation!")

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

    print(f"[INFO] Loaded class ID range (0-based): min={labels.min()}, max={labels.max()}")

    return video_paths, labels, video_names



def extract_frame_probs_from_video(model, video_path, frame_step=5):
    """
    Sample frames from a video and run YOLO11 classification.
    Returns:
        frame_probs -> shape [T, C], where T=number of sampled frames, C=number of classes
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return None

    frame_probs = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            # YOLO forward inference
            results = model(frame, verbose=False)
            # Classification task: results[0].probs.data is softmax probabilities
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
    Convert [T, C] frame probability matrix into a single video feature vector.
    Uses mean / max / std statistics, concatenated -> [3C]
    """
    mean_feat = frame_probs.mean(axis=0)   # [C]
    max_feat = frame_probs.max(axis=0)     # [C]
    std_feat = frame_probs.std(axis=0)     # [C]

    video_feat = np.concatenate([mean_feat, max_feat, std_feat], axis=0)  # [3C]
    return video_feat


def main():
    # 1. Load YOLO11 classification model
    print("[INFO] Loading YOLO11 classification model...")
    model = YOLO(YOLO_WEIGHTS)

    # 2. Load CSV and retrieve video list + labels
    print("[INFO] Reading CSV and loading video list + labels...")
    video_paths, labels, video_names = load_video_paths_and_labels(CSV_PATH, VIDEO_ROOT)
    print(f"[INFO] Number of valid videos: {len(video_paths)}")

    all_feats = []
    valid_labels = []
    valid_names = []

    # 3. Iterate over each video: sampling + probability extraction
    for v_path, label, name in tqdm(
        list(zip(video_paths, labels, video_names)),
        total=len(video_paths),
        desc="Extracting video features"
    ):
        frame_probs = extract_frame_probs_from_video(model, v_path, frame_step=FRAME_STEP)

        if frame_probs is None:
            print(f"[WARN] Empty frame feature, skipping: {v_path}")
            continue

        video_feat = aggregate_video_features(frame_probs)  # [3C]
        all_feats.append(video_feat)
        valid_labels.append(label)
        valid_names.append(name)

    if len(all_feats) == 0:
        print("[ERROR] No video features extracted. Check video paths and CSV configuration.")
        return

    all_feats = np.stack(all_feats, axis=0)           # [N, 3C]
    valid_labels = np.array(valid_labels, dtype=np.int64)
    valid_names = np.array(valid_names)

    # 4. Save feature arrays
    np.save(OUT_FEATURES_PATH, all_feats)
    np.save(OUT_LABELS_PATH, valid_labels)
    np.save(OUT_VIDEO_NAMES_PATH, valid_names)

    print(f"[INFO] Features saved to: {OUT_FEATURES_PATH}, shape={all_feats.shape}")
    print(f"[INFO] Labels saved to: {OUT_LABELS_PATH}, shape={valid_labels.shape}")
    print(f"[INFO] Video names saved to: {OUT_VIDEO_NAMES_PATH}, shape={valid_names.shape}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
