#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from joblib import dump  
from sklearn.decomposition import PCA


FEATURES_PATH = "video_features.npy"
LABELS_PATH = "video_labels.npy"

TEST_SIZE = 0.2          # 验证集比例
RANDOM_STATE = 42        # 随机种子，保证可复现

BALANCE_MODE = "undersample"
# 选项：
#   "undersample"：对多数类做欠采样，所有类都减到最少类的数量（最安全）
#   "oversample"：对少数类做过采样，所有类都增到最多类的数量（可能过拟合）

XGB_PARAMS = dict(
    objective="multi:softmax",
    eval_metric="mlogloss",

    # 稍微放宽一点
    n_estimators=500,    
    max_depth=3,         

    learning_rate=0.05,

    subsample=0.4,       
    colsample_bytree=0.4,

    reg_lambda=3.0,      
    reg_alpha=0.5,       
    min_child_weight=3,  
    gamma=0.5,           

    tree_method="hist",
)


# ==========================================


def make_class_balanced(X, y, mode="undersample", random_state=42):
    """
    对 (X, y) 做类别均衡：
        - undersample：多数类欠采样到“最少类”的数量
        - oversample：少数类过采样到“最多类”的数量
    返回均衡后的 (X_bal, y_bal)
    """
    rng = np.random.RandomState(random_state)
    classes = np.unique(y)

    # 统计每个类别的索引
    indices_by_class = {c: np.where(y == c)[0] for c in classes}
    counts = {c: len(idx) for c, idx in indices_by_class.items()}

    print("[INFO] 原始训练集各类别样本数：")
    for c in sorted(classes):
        print(f"  class {c}: {counts[c]} samples")

    if mode == "undersample":
        target_n = min(counts.values())
        print(f"[INFO] 使用欠采样（undersample），每类目标样本数 = {target_n}")
        new_indices = []
        for c, idx in indices_by_class.items():
            if len(idx) > target_n:
                chosen = rng.choice(idx, size=target_n, replace=False)
            else:
                chosen = idx
            new_indices.append(chosen)

    elif mode == "oversample":
        target_n = max(counts.values())
        print(f"[INFO] 使用过采样（oversample），每类目标样本数 = {target_n}")
        new_indices = []
        for c, idx in indices_by_class.items():
            if len(idx) < target_n:
                extra = rng.choice(idx, size=target_n - len(idx), replace=True)
                chosen = np.concatenate([idx, extra], axis=0)
            else:
                chosen = idx
            new_indices.append(chosen)

    else:
        raise ValueError("mode 必须是 'undersample' 或 'oversample'")

    new_indices = np.concatenate(new_indices, axis=0)
    rng.shuffle(new_indices)

    X_bal = X[new_indices]
    y_bal = y[new_indices]

    # 再统计一次
    classes_bal = np.unique(y_bal)
    counts_bal = {c: (y_bal == c).sum() for c in classes_bal}
    print("[INFO] 均衡后训练集各类别样本数：")
    for c in sorted(classes_bal):
        print(f"  class {c}: {counts_bal[c]} samples")

    return X_bal, y_bal


def main():
    # 1. 读取特征和标签
    X = np.load(FEATURES_PATH)   # [N, D]
    y = np.load(LABELS_PATH)     # [N]

    print(f"[INFO] 特征维度: {X.shape}, 标签数量: {y.shape}")
    classes = np.unique(y)
    num_classes = len(classes)
    print(f"[INFO] 类别数: {num_classes}, 类别 id 列表: {classes.tolist()}")

    # 2. 划分训练集 / 验证集（按类别比例 stratify）
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"[INFO] 训练集大小: {X_train.shape[0]}, 验证集大小: {X_val.shape[0]}")

    # 3. 仅对“训练集”做类别均衡（验证集保持真实分布）
    X_train_bal, y_train_bal = make_class_balanced(
        X_train, y_train,
        mode=BALANCE_MODE,
        random_state=RANDOM_STATE
    )

    # 4. 训练 XGBoost
    print("\n[INFO] 开始训练 XGBoost（使用均衡后的训练集）...")
    xgb_model = XGBClassifier(
    num_class=num_classes,
    random_state=RANDOM_STATE,
    **XGB_PARAMS)

    # 只对训练集 fit
    pca = PCA(n_components=0.95)  # 保留 95% 方差
    X_train_pca = pca.fit_transform(X_train_bal)
    X_val_pca = pca.transform(X_val)
    
    # 这里 eval_set 只是为了能看到每轮在 val 上的 mlogloss 变化
    xgb_model.fit(
        X_train_bal,
        y_train_bal,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    

    # 5. 评估：在均衡训练集上 & 原验证集上
    y_train_pred = xgb_model.predict(X_train_bal)
    train_acc = accuracy_score(y_train_bal, y_train_pred)
    print(f"\n[XGBoost] Balanced Train Acc: {train_acc:.4f}")

    y_val_pred = xgb_model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"[XGBoost] Validation Acc:      {val_acc:.4f}")

    print("\n[INFO] 验证集详细分类报告：")
    print(classification_report(y_val, y_val_pred, digits=4))

    dump(xgb_model, "xgb_video_cls.pkl")
    print("[INFO] XGBoost 模型已保存为 xgb_video_cls.pkl")


if __name__ == "__main__":
    main()
