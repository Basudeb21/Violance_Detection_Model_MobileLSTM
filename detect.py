import os
import cv2
import numpy as np
import redis
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.models import load_model

# ─────────────────────────────────────────────
# PATH & MODEL SETUP
# ─────────────────────────────────────────────
from insert_violation_db import save_video_violation_to_db  # DB helper



model = load_model("model/MobBiLSTM_model_saved101.keras")

# ─────────────────────────────────────────────
# REDIS SETUP
# ─────────────────────────────────────────────
r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
REDIS_QUEUE = "fliqz_moderation_image_video_queue"

# ─────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────
FRAME_SIZE = (64, 64)
FRAMES_PER_CLIP = 16

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, FRAME_SIZE)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames found in video: {video_path}")

    if len(frames) < FRAMES_PER_CLIP:
        frames.extend([frames[-1]] * (FRAMES_PER_CLIP - len(frames)))
    else:
        frames = frames[:FRAMES_PER_CLIP]

    frames = np.array(frames) / 255.0
    return np.expand_dims(frames, axis=0)

def preprocess_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Image not found: {image_path}")
    frame = cv2.resize(frame, FRAME_SIZE)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # replicate the image to mimic a video clip
    frames = np.array([frame] * FRAMES_PER_CLIP) / 255.0
    return np.expand_dims(frames, axis=0)

def predict_violation(file_path, file_type='video'):
    if file_type == 'video':
        clip = preprocess_video(file_path)
    elif file_type == 'image':
        clip = preprocess_image(file_path)
    else:
        raise ValueError("file_type must be 'video' or 'image'")

    pred_val = model.predict(clip)[0][0]
    violation = 0 if pred_val >= 0.5 else 1
    return violation, pred_val

def determine_file_type(filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext in ['.mp4', '.avi', '.mov', '.mkv']:
        return 'video'
    elif ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        return 'image'
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# ─────────────────────────────────────────────
# REDIS PROCESSOR
# ─────────────────────────────────────────────
def process_redis_queue():
    print("🎧 Listening to Redis queue for files...")
    while True:
        try:
            _, message = r.blpop(REDIS_QUEUE)
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                print("Invalid JSON:", message)
                continue

            filename = data.get("filename")
            if not filename:
                print("❌ No filename in message")
                continue

            try:
                file_type = determine_file_type(filename)
            except ValueError as e:
                print("❌", e)
                continue

            violation, pred_val = predict_violation(filename, file_type=file_type)
            print(f"Prediction for {file_type}: {'Violent' if violation else 'Non-violent'} ({pred_val:.4f})")

            # Save result to DB
            save_video_violation_to_db(data, violation)

        except Exception as e:
            print("❌ Error processing Redis item:", repr(e))

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    process_redis_queue()
