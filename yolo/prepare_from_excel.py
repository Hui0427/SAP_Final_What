import cv2
import pandas as pd
from pathlib import Path

# ======================= 配置区（根据你自己改） =======================

# Excel 标注文件路径
EXCEL_PATH = Path(r"E:/A_KCL/Sensing and perception/dataset/annotations/train_set_labels.xlsx")

# Excel 列名（如果是第一列、第二列、第三列没有列名，可以自己在 Excel 里加上表头）
COL_VIDEO_NAME = "video_name"      # 第 1 列：视频名（不含扩展名）
COL_CLASS_NAME = "class_name_en"   # 第 2 列：动作类别英文描述
COL_CLASS_ID = "class_id"          # 第 3 列：类别序号（可选，主要用来参考）

# 训练 / 测试视频所在的根目录
TRAIN_VIDEO_ROOT = Path(r"E:/A_KCL/Sensing and perception/dataset/train_set")   # 里面是 .avi 等视频
TEST_VIDEO_ROOT = Path(r"E:/A_KCL/Sensing and perception/dataset/test_set")

# 视频扩展名（如果不全是 .avi，可以改成列表逻辑，或者在下面的 find_video_file 里改）
VIDEO_EXT = ".avi"

# 输出的数据集根目录（会自动创建）
DATASET_ROOT = Path("dataset")

# 抽帧相关配置
FRAME_STEP_TRAIN = 2        # train：每隔多少帧取一张
FRAME_STEP_TEST = 2         # test：每隔多少帧取一张
MIN_FRAMES_PER_VIDEO = 5    # 至少保存多少帧（防止超短视频）

# test 没有标签，就统一放在这个子目录下
TEST_SUBDIR_NAME = "unlabeled"

# =====================================================================


def find_video_file(root: Path, video_stem: str, ext: str = VIDEO_EXT) -> Path | None:
    """
    在 root 目录下查找名字为 video_stem + ext 的视频文件。
    如果找不到，可以按需改成更复杂的搜索（例如大小写忽略、多个扩展名等）
    """
    candidate = root / f"{video_stem}{ext}"
    if candidate.exists():
        return candidate

    # 如果严格名字找不到，也可以尝试模糊搜索一下（比如不同大小写）
    # 这里给一个简单示例：在整个 root 下找同名 stem 的任意视频
    for p in root.rglob("*"):
        if p.is_file() and p.stem == video_stem:
            return p

    return None


def extract_frames(video_path: Path, out_dir: Path, frame_step: int):
    """
    从一个视频中按 frame_step 抽帧保存到 out_dir
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return

    frame_idx = 0
    save_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            save_path = out_dir / f"{video_path.stem}_f{save_idx:04d}.jpg"
            cv2.imwrite(str(save_path), frame)
            save_idx += 1

        frame_idx += 1

    cap.release()

    if save_idx < MIN_FRAMES_PER_VIDEO:
        print(f"[INFO] {video_path} only saved {save_idx} frames (< {MIN_FRAMES_PER_VIDEO}).")


def process_train_from_excel():
    """
    使用 Excel 中的标注信息，为 train 集创建目录并抽帧
    目录结构：dataset/train/<class_name>/xxx_f0000.jpg
    """
    print("=== 处理 train 数据（来自 Excel 标注） ===")

    df = pd.read_excel(EXCEL_PATH)

    # 简单检查列是否存在
    for col in [COL_VIDEO_NAME, COL_CLASS_NAME, COL_CLASS_ID]:
        if col not in df.columns:
            raise ValueError(f"Excel 中找不到列: {col}, 请检查列名或修改脚本配置。")

    # 遍历每一行
    for idx, row in df.iterrows():
        video_stem = str(row[COL_VIDEO_NAME]).strip()
        class_name = str(row[COL_CLASS_NAME]).strip()
        class_id = row[COL_CLASS_ID]

        if not video_stem:
            print(f"[WARN] 第 {idx} 行视频名为空，跳过。")
            continue

        video_path = find_video_file(TRAIN_VIDEO_ROOT, video_stem)
        if video_path is None:
            print(f"[WARN] 找不到 train 视频文件: {video_stem} (第 {idx} 行)")
            continue

        # 以类别英文描述作为子目录名，也可以改成 f"{class_id:02d}_{class_name}"
        out_dir = DATASET_ROOT / "train" / class_name

        print(f"[TRAIN] {video_path.name} -> class='{class_name}' (id={class_id}), 抽帧到: {out_dir}")
        extract_frames(video_path, out_dir, FRAME_STEP_TRAIN)


def process_test_unlabeled():
    """
    为 test 数据集抽帧（无标签），统一放到 dataset/test/unlabeled/
    """
    print("=== 处理 test 数据（无标签） ===")

    out_dir_base = DATASET_ROOT / "test" / TEST_SUBDIR_NAME
    out_dir_base.mkdir(parents=True, exist_ok=True)

    # 简单地把 TEST_VIDEO_ROOT 下的所有视频都抽帧
    video_files = list(TEST_VIDEO_ROOT.rglob(f"*{VIDEO_EXT}"))
    if not video_files:
        print(f"[WARN] 在 {TEST_VIDEO_ROOT} 下找不到任何扩展名为 {VIDEO_EXT} 的视频。")
        return

    for video_path in video_files:
        # 可以按视频名单独建立子目录，也可以直接全部放在同一个文件夹
        # 这里选择直接都放在一个目录中，文件名中包含 stem，方便区分
        print(f"[TEST] 抽帧: {video_path} -> {out_dir_base}")
        extract_frames(video_path, out_dir_base, FRAME_STEP_TEST)


def main():
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    process_train_from_excel()
    process_test_unlabeled()

    print("\n=== 完成！数据集目录位于:", DATASET_ROOT.resolve(), "===")


if __name__ == "__main__":
    main()
