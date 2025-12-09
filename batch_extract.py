import os
import glob
import numpy as np
import cv2
import mediapipe as mp

# --- Configuration ---
TRAIN_FOLDER = "train_set"  
TEST_FOLDER = "test_set"    
OUTPUT_FOLDER = "skeleton_data"
OVERWRITE = True  # ⬅️ If True, re-extract and overwrite existing files; if False, skip existing ones

# --- Initialize MediaPipe ---
print("🔧 Initializing MediaPipe Pose model...")
mp_pose = mp.solutions.pose
# model_complexity=1 is a balanced option. If your computer is fast, you may change to 2 (higher accuracy but slower)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

def process_one_file(video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): 
        print(f"❌ Unable to open video: {video_path}")
        return False
    
    frames_data = []
    
    # 🧠 [Key Improvement] Store previous valid frame data
    # Initialize with zeros in case the very first frame contains no person
    last_valid_frame = [0] * 132 
    has_valid_frame = False

    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        frame_landmarks = []
        
        if results.pose_landmarks:
            # ✅ Person detected
            for lm in results.pose_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
            
            # Update last valid frame data
            last_valid_frame = frame_landmarks
            has_valid_frame = True
            frames_data.append(frame_landmarks)
        else:
            # ❌ No detection (occlusion or missing frame)
            # Use last valid frame instead of zeros (Forward Fill)
            if has_valid_frame:
                frames_data.append(last_valid_frame)
            else:
                # If video begins with no detection, fill with zeros
                frames_data.append([0] * 132)
        
        frame_count += 1
    
    cap.release()
    
    # Save as .npy
    if len(frames_data) > 0:
        np.save(save_path, np.array(frames_data))
        return True
    else:
        return False

def run_batch(folder_name, split_type):
    # Search for videos
    search_pattern = os.path.join(folder_name, "*.avi") # If your videos are mp4, change to "*.mp4"
    files = glob.glob(search_pattern)
    print(f"\n📂 [{split_type}] Found {len(files)} video files")
    
    if len(files) == 0:
        print("⚠️ Warning: The folder is empty!")
        return

    # Create output folder
    save_dir = os.path.join(OUTPUT_FOLDER, split_type)
    os.makedirs(save_dir, exist_ok=True)
    
    print("🚀 Starting extraction...")
    count = 0
    for i, video_path in enumerate(files):
        file_id = os.path.splitext(os.path.basename(video_path))[0]
        save_path = os.path.join(save_dir, file_id + ".npy")
        
        # Skip if needed
        if os.path.exists(save_path) and not OVERWRITE:
            print(f"\r[Skipped] {file_id}.npy already exists", end="")
            continue

        # Progress display
        print(f"\r[{i+1}/{len(files)}] Processing: {file_id} ... ", end="")
        
        success = process_one_file(video_path, save_path)
        if success: 
            count += 1
            
    print(f"\n🎉 {split_type} extraction completed! Successfully processed {count} files.")

# --- Execute ---
if __name__ == "__main__":
    print("-" * 30)
    print(f"Current working directory: {os.getcwd()}")
    print("-" * 30)

    if os.path.exists(TRAIN_FOLDER):
        run_batch(TRAIN_FOLDER, "train")
    else:
        print(f"❌ Training folder not found: {TRAIN_FOLDER}")

    if os.path.exists(TEST_FOLDER):
        run_batch(TEST_FOLDER, "test")
    else:
        print(f"❌ Test folder not found: {TEST_FOLDER}") # Test folder may not exist; not a critical error
        

    print("\n🏁 All processing finished. Please re-run the training script!")
