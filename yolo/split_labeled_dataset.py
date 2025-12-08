import random
from pathlib import Path
import shutil

# ================== 配置区 ==================

# 带标签数据集的根目录
DATASET_ROOT = Path("dataset")

# 原来所有有标签的图片都在这里：dataset/train/<class>/
TRAIN_DIR_NAME = "train"

# 新建一个验证集目录：dataset/val/<class>/
VAL_DIR_NAME = "val"

# 训练集比例（每个类别内部都按这个比例划分）
TRAIN_RATIO = 0.8

# 支持的图片后缀
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]

# 随机种子，保证每次划分一致
RANDOM_SEED = 42

# ===========================================


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def main():
    random.seed(RANDOM_SEED)

    train_root = DATASET_ROOT / TRAIN_DIR_NAME
    val_root = DATASET_ROOT / VAL_DIR_NAME

    if not train_root.exists():
        raise FileNotFoundError(f"找不到训练集目录: {train_root}")

    val_root.mkdir(parents=True, exist_ok=True)

    # 遍历每个类别文件夹
    for class_dir in train_root.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        print(f"\n===== 处理类别: {class_name} =====")

        # 找到该类别下的所有图片
        images = [p for p in class_dir.iterdir() if p.is_file() and is_image_file(p)]
        if not images:
            print(f"[WARN] 类别 {class_name} 没有图片，跳过。")
            continue

        print(f"该类别共有 {len(images)} 张图片")

        # 打乱顺序
        random.shuffle(images)

        # 计算划分点
        n_total = len(images)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = n_total - n_train

        train_imgs = images[:n_train]
        val_imgs = images[n_train:]

        print(f"划分结果: train={len(train_imgs)}, val={len(val_imgs)}")

        # 为该类别创建 val 子目录
        val_class_dir = val_root / class_name
        val_class_dir.mkdir(parents=True, exist_ok=True)

        # 把 val 的图片从 dataset/train/<class>/ 移动到 dataset/val/<class>/
        for img_path in val_imgs:
            target_path = val_class_dir / img_path.name
            print(f"  移动: {img_path.name} -> {target_path}")
            shutil.move(str(img_path), str(target_path))

    print("\n=== 划分完成！请检查目录结构：")
    print(f"训练集: {train_root.resolve()}")
    print(f"验证集: {val_root.resolve()}")


if __name__ == "__main__":
    main()
