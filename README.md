# SAP Final Project – Two-Stream Action Recognition for Industrial HRI

## 1. Project Overview

This project tackles the **HRI30 industrial human–robot interaction dataset**.
Our goal is to **classify 30 industrial actions** (e.g. *DeliverObject, InspectObject, PickUpTool, PlaceTool* …) from video, in the context of human–robot collaboration.

We design and implement a **two-stream (multi-modal) architecture**:

* **Skeleton Stream**: a lightweight **CNN-LSTM + Attention** model trained on 3D human pose sequences extracted with MediaPipe.
* **RGB Stream**: a 3D CNN (**R2Plus1D-18**) trained on raw video clips.
* **Late Fusion**: per-class probability fusion
  [
  p_{\text{final}} = \alpha , p_{\text{RGB}} + (1 - \alpha), p_{\text{Skeleton}}
  ]
  with **α = 0.8** (RGB-dominant, skeleton as a robust auxiliary cue).

In addition, we implement a **YOLO-based pipeline** as a **second independent approach**:

* Run YOLO on each frame to get bounding boxes and object categories.
* Aggregate detections temporally to produce sequence-level features.
* Train a classifier on top of YOLO features to predict the action class.
* Export a separate CSV of predictions for comparison and as a backup submission.

The final submission to the coursework consists of **CSV predictions** generated from:

1. The **two-stream fusion model**.
2. The **YOLO-based pipeline**.

---

## 2. How to Run (Two-Stream Pipeline)

> This section focuses on the **two-stream (Skeleton + RGB) model**, which is our main solution.

### 2.1. Environment & Dependencies

We recommend using **Python 3.10+** and `conda` / `venv`.

Key dependencies:

* `torch`, `torchvision`
* `numpy`, `pandas`, `opencv-python`
* `mediapipe`
* `matplotlib`, `seaborn`
* `scikit-learn`

Example (conda):

```bash
conda create -n sap310 python=3.10
conda activate sap310

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # or CPU version
pip install numpy pandas opencv-python mediapipe matplotlib seaborn scikit-learn
```

Make sure the dataset is placed as:

```bash
sap_dataset/
├── train_set/              # *.avi training videos
├── test_set/               # *.avi test videos
└── annotations/
    └── train_set_labels.csv
```

And the project folder (this repo) contains:

```bash
SAP_Group_What/
├── batch_extract.py
├── train_model_v3.py
├── rgb_model.py
├── predict_multimodel_final.py
├── analyze_fusion.py
├── skeleton_data/          # will be created by batch_extract.py
├── train_set/              # (optional: symlink or copy from sap_dataset)
├── test_set/               # (optional: symlink or copy from sap_dataset)
└── annotations/
    └── train_set_labels.csv
```

You can either:

* Copy/symlink `train_set/`, `test_set/`, `annotations/` into this folder, **or**
* Edit the paths inside the scripts to point to your dataset location.

---

### 2.2. Step 1 – Extract Skeletons with MediaPipe

Script: **`batch_extract.py`**

This script:

* Loads each `*.avi` video from `train_set/` and `test_set/`.
* Runs **MediaPipe Pose** on all frames.
* Handles **occlusion / missing detections** using **forward fill**:

  * If a frame has no detection, it copies the last valid frame’s 132-dim skeleton.
* Saves per-video skeleton sequences as `*.npy` files.

Run:

```bash
python batch_extract.py
```

Expected folder structure after running:

```bash
skeleton_data/
├── train/
│   ├── CID01_SID01_VID01.npy
│   ├── ...
└── test/
    ├── testvideo001.npy
    ├── ...
```

---

### 2.3. Step 2 – Train Skeleton Model (CNN-LSTM + Attention)

Script: **`train_model_v3.py`**

This script:

* Loads skeleton sequences from `skeleton_data/train/`.
* Normalizes skeletons with:

  * **Root centering** at hip center.
  * **Shoulder-width scaling** for scale invariance.
* Applies light **data augmentation** (Gaussian noise, random hand occlusion).
* Trains a **Lightweight CNN-LSTM with Multi-head Attention**.
* Uses **class weights** to handle class imbalance.
* Tracks validation accuracy and saves the best model as **`best_model_v3.pth`**.

Run:

```bash
python train_model_v3.py
```

After training, you should have:

```bash
best_model_v3.pth
```

---

### 2.4. Step 3 – Train RGB Model (R2Plus1D-18)

Script: **`rgb_model.py`**

This script:

* Splits `train_set` into train/validation (80/20).
* For each video:

  * Resizes frames to **128×128**.
  * Uniformly samples **16 frames** per clip.
  * Converts BGR→RGB.
  * Normalizes with **Kinetics-400 mean/std**.
* Loads **`r2plus1d_18` pretrained on Kinetics-400**.
* Fine-tunes the final classification layer for 30 classes.
* Uses **mixed-precision + gradient accumulation** for efficiency.
* Saves the best RGB model as **`best_model_rgb.pth`**.

Run:

```bash
python rgb_model.py
```

After training, you should have:

```bash
best_model_rgb.pth
```

---

### 2.5. Step 4 – Generate Final Submission CSV (Two-Stream Fusion)

Script: **`predict_multimodel_final.py`**

This script:

1. Loads **label list** from `annotations/train_set_labels.csv`.
2. Loads:

   * `best_model_v3.pth`  (skeleton model)
   * `best_model_rgb.pth` (RGB model)
3. For each **test video** in `test_set/`:

   * Loads its skeleton from `skeleton_data/test/`.
   * Runs the **skeleton stream** → probability vector (p_{\text{Skeleton}}).
   * Runs the **RGB stream** → probability vector (p_{\text{RGB}}).
   * Applies **late fusion**:
     [
     p_{\text{final}} = 0.8 , p_{\text{RGB}} + 0.2 , p_{\text{Skeleton}}
     ]
   * Takes `argmax` over `p_final` to get the final action label.
4. Saves output as **`test_set_labels_fusion.csv`** with format:

```text
CID01_SID01_VID01.avi,DeliverObject
CID01_SID01_VID02.avi,DeliverObject
...
```

Run:

```bash
python predict_multimodel_final.py
```

This CSV is the **two-stream submission file**.

---

### 2.6. (Optional) Local Validation & Confusion Matrix

Script: **`analyze_fusion.py`**

This script:

* Uses **only the training set**.
* Randomly selects **20%** as a **pseudo-validation set**.
* Runs skeleton, RGB, and fusion prediction on this split.
* Computes:

  * Skeleton-only accuracy.
  * RGB-only accuracy.
  * Fusion accuracy.
* Plots a **confusion matrix** and saves `fusion_analysis.png`.

Run:

```bash
python analyze_fusion.py
```

This is useful for the **report**, to show that **fusion improves accuracy**.

---

## 3. Our Design Choices & Pipeline Explanation

### 3.1. Why a Two-Stream Architecture?

The HRI30 dataset contains:

* People **performing actions** (gesture/motion patterns).
* **Tools and objects** being used (drills, polishers, components).
* Industrial settings with **occlusions, lighting changes, and clutter**.

A single modality is not enough:

* **Skeleton only**:

  * Pros: robust to lighting, background clutter, appearance.
  * Cons: cannot “see” which object is being manipulated (e.g. different tools may have similar hand motion).
* **RGB only**:

  * Pros: can see **objects, tools, environment cues**.
  * Cons: more sensitive to illumination, camera angle, and occlusion.

So we build a **two-stream model** where:

* The **RGB stream** is the **primary** predictor (*α = 0.8*).
* The **skeleton stream** corrects some mistakes, especially when object appearance is ambiguous or the pose is very distinctive.

This combination gives us:

* High **overall accuracy** on the test set.
* Better **robustness** across different subjects and camera positions.

---

### 3.2. Skeleton Stream – Lightweight CNN-LSTM with Attention

**Core script**: `train_model_v3.py`
**Model class**: `LightweightCNNLSTM`

**Input**:
Per-video skeleton sequence from `batch_extract.py`:

* Generated with **MediaPipe Pose**, 33 keypoints × 4 values (x, y, z, visibility).
* We only use the **XYZ** coordinates (drop visibility) → 33 × 3 = **99 features per frame**.
* Sequence length is **padded or centered to 100 frames**.

**Preprocessing & Normalization** (`normalize_skeleton` logic):

1. **Root centering**:

   * Hip center = average of left and right hip joints (IDs 23 and 24).
   * Subtract this root from all joints → translation invariance.
2. **Shoulder-width scaling**:

   * Distance between left (11) and right (12) shoulders.
   * Divide all joint coordinates by this distance → scale invariance.

**Augmentation** (in `SkeletonDatasetV8.apply_augmentation`):

* Add small Gaussian noise to joint positions.
* Randomly zero out one hand (left or right) to simulate occlusions.

**Model architecture**:

* **1D CNN** over the time axis:

  * Conv1d(99 → 64), kernel size 3, padding 1.
  * No pooling, to preserve sequence length.
* **Bi-LSTM**:

  * Input = 64, hidden size = 128, 2 layers, bidirectional.
* **Multi-Head Attention**:

  * Over the LSTM output, with 4 heads.
* **Mean temporal pooling** + final **Linear** layer to 30 classes.

**Training**:

* Loss: `CrossEntropyLoss` with **class weights** (`compute_class_weight`) to handle imbalance.
* Optimizer: `Adam`.
* LR scheduler: `ReduceLROnPlateau` on validation accuracy.
* Output checkpoint: `best_model_v3.pth`.

We also ran a number of **ablation / scratch experiments** (`multimodel_scratch.py`, `train_rgb_scratch.py`) to show that:

* Training RGB from scratch performs significantly worse.
* Skeleton stream alone achieves around **70%** accuracy.
* Pretrained RGB + well-designed skeleton fusion performs best.

---

### 3.3. RGB Stream – R2Plus1D-18 on Raw Video

**Core script**: `rgb_model.py`

**Input pipeline**:

* For each `.avi`:

  * Read all frames via OpenCV.
  * Resize each frame to `128 × 128`.
  * Uniformly sample **16 frames** over the whole video.
  * Stack to tensor `(C, T, H, W) = (3, 16, 128, 128)`.

**Normalization**:

* Divide by 255 → in `[0,1]`.
* Subtract **Kinetics-400 mean = [0.432, 0.394, 0.376]**.
* Divide by **Kinetics-400 std = [0.228, 0.221, 0.217]**.

**Model**:

* `torchvision.models.video.r2plus1d_18` with Kinetics-400 pretrained weights.
* Replace `fc` with `nn.Linear(fc.in_features, 30)`.

**Training tricks**:

* **Mixed precision** with `torch.cuda.amp.autocast` + `GradScaler`.
* **Gradient accumulation** (e.g. `ACCUMULATION_STEPS = 4`) to reduce memory footprint.
* Optimizer: SGD + momentum.
* Scheduler: `ReduceLROnPlateau` on validation accuracy.

This stream alone achieves around **90%+ accuracy** on the validation split.

---

### 3.4. Late Fusion – Two-Stream Combination

**Core script**: `predict_multimodel_final.py`, `analyze_fusion.py`

For each sample:

1. Skeleton stream → logits_s → probs_s = softmax(logits_s).
2. RGB stream → logits_r → probs_r = softmax(logits_r).
3. **Fusion**:
   [
   p_{\text{final}} = 0.8 , p_{\text{RGB}} + 0.2 , p_{\text{Skeleton}}
   ]
4. Final prediction = `argmax(p_final)`.

We also implemented a **grid search** version (`multimodel_scratch.py`) that evaluates:

* Skeleton-only accuracy.
* RGB-only accuracy.
* Fusion accuracy for **α ∈ [0,1]**.

This shows that:

* With pretrained RGB, the **optimal α is high for RGB**.
* When RGB is trained from scratch (weaker), the **optimal α shifts towards skeleton**.

This supports our argument in the report:

> “The skeleton stream acts as a strong complementary modality, especially when RGB is unreliable or under-trained.”

---

## 4. YOLO-Based Pipeline

> In addition to the two-stream model, we implemented a **YOLO-based action recognition pipeline**, which also produced a high-accuracy CSV (`video_predictions.csv`). This serves as a **second strong baseline** and demonstrates our ability to explore alternative solutions.

Code under: **`yolo/`** (in the GitHub repo), plus a final CSV (e.g. `video_predictions.csv`).

### 4.1. High-Level Idea

Instead of modeling full video with 3D CNN, we:

1. Run **YOLO** on each frame:

   * Detect human(s), tools, manipulated objects.
2. Extract **per-frame features**:

   * Object categories/labels.
   * Bounding box positions / sizes.
   * Possibly the relative position between hand and tools.
3. Aggregate features over time:

   * E.g. histogram of detected objects.
   * Temporal statistics (max/mean count, co-occurrence patterns).
4. Train a classifier (e.g. MLP, shallow network, or even rule-based mapping) to predict the **action label** for each video.

This is particularly good at:

* Distinguishing actions that mainly differ by **which object** is being used (e.g. *UseDrill* vs *UsePolisher*).
* Working as a **fast alternative** when skeleton extraction or heavy 3D CNNs are too expensive.

### 4.2. What the YOLO Code Does (Conceptually)

In the `yolo/` folder (see GitHub):

* **YOLO inference script**:

  * Loads each video, runs YOLO on all (or sampled) frames.
  * Saves detected objects per frame (class IDs, confidences, boxes).
* **Feature aggregation script**:

  * Converts frame-level detections into video-level features (counts, types, co-occurrence).
* **Classifier**:

  * Trains a model on these features + labels from `train_set_labels.csv`.
  * Runs prediction on test videos to produce **`video_predictions.csv`** in the format required by the coursework.

This YOLO solution:

* Gives an **independent CSV submission** (YOLO-based prediction).
* Strengthens the report by:

  * Comparing **YOLO vs RGB vs Skeleton vs Fusion**.
  * Showing that **object-centric** reasoning is highly effective in an industrial HRI setting.

---

## 5. Summary

* Our **main solution** is a **Two-Stream Architecture** (Skeleton + RGB) with **late fusion**, implemented in:

  * `batch_extract.py`
  * `train_model_v3.py`
  * `rgb_model.py`
  * `predict_multimodel_final.py`
  * `analyze_fusion.py`
* We carefully designed:

  * Pose normalization (root centering + shoulder scaling).
  * Lightweight yet expressive CNN-LSTM-Attention for skeletons.
  * Pretrained R2Plus1D-18 for RGB.
  * Fusion weight selection and validation.
* We also implemented a **YOLO-based pipeline** as an additional high-accuracy method and as a strong baseline.

For the coursework submission, we provide:

1. **Two-Stream Fusion CSV** (`test_set_labels_fusion.csv`).
2. **YOLO-Based CSV** (`video_predictions.csv`).
3. Code and analysis scripts for training, fusion, and evaluation.

This setup gives a **reproducible, multi-modal, and well-analyzed** solution to the SAP final project.
