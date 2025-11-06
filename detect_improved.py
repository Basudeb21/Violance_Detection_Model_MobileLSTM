import os
import cv2
import numpy as np
import redis
import json
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from skimage.transform import resize

# ─────────────────────────────────────────────
# PATH & MODEL SETUP
# ─────────────────────────────────────────────
from insert_violation_db import save_video_violation_to_db  # DB helper

# ─────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────
def mamon_videoFightModel2(weight='mamonbest947oscombo-drive.h5'):
    layers = tf.keras.layers
    models = tf.keras.models
    num_classes = 2

    base_model = tf.keras.applications.vgg19.VGG19(
        include_top=False,
        weights='imagenet',
        input_shape=(160, 160, 3)
    )

    cnn = models.Sequential([
        base_model,
        layers.Flatten()
    ])

    model = models.Sequential([
        layers.TimeDistributed(cnn, input_shape=(30, 160, 160, 3)),
        layers.LSTM(30, return_sequences=True),
        layers.TimeDistributed(layers.Dense(90)),
        layers.Dropout(0.1),
        layers.GlobalAveragePooling1D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='sigmoid')
    ])

    adam = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.load_weights(weight)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

# Load model once at startup
model = mamon_videoFightModel2(weight='mamonbest947oscombo-drive.h5')

# ─────────────────────────────────────────────
# REDIS SETUP
# ─────────────────────────────────────────────
r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
REDIS_QUEUE = "fliqz_moderation_image_video_queue"

# ─────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────
FRAMES_PER_CLIP = 30
FRAME_SIZE = (160, 160)
ACCURACY_THRESHOLD = 0.65

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def preprocess_video(video_path):
    """Reads the full video and returns a list of 30-frame clips."""
    vc = cv2.VideoCapture(video_path)
    
    if not vc.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")
    
    all_frames = []

    # Read ALL frames
    while True:
        rval, frame = vc.read()
        if not rval:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frm = resize(frame, (160, 160, 3))
        if np.max(frm) > 1:
            frm = frm / 255.0
        all_frames.append(frm)
    vc.release()

    if len(all_frames) == 0:
        raise ValueError(f"No frames found in video: {video_path}")

    # Split into 30-frame chunks
    clips = []
    for i in range(0, len(all_frames), 30):
        chunk = all_frames[i:i + 30]
        if len(chunk) < 30:
            # pad the last chunk if shorter than 30
            pad_len = 30 - len(chunk)
            chunk += [chunk[-1]] * pad_len  # Use last frame instead of zeros
        clips.append(np.array(chunk))

    return np.array(clips)  # shape: (num_clips, 30, 160, 160, 3)

def preprocess_image(image_path):
    """Reads an image and creates a 30-frame clip by replicating it."""
    frame = cv2.imread(image_path)
    
    if frame is None:
        raise ValueError(f"Unable to load image: {image_path}")
    
    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize and normalize
    frm = resize(frame, (160, 160, 3))
    if np.max(frm) > 1:
        frm = frm / 255.0
    
    # Create 30 identical frames
    frames = np.array([frm] * 30)
    
    return np.expand_dims(frames, axis=0)  # shape: (1, 30, 160, 160, 3)

def predict_violation(file_path, file_type='video'):
    """
    Unified function to predict violence in video or image.
    
    Args:
        file_path: Path to video or image file
        file_type: 'video' or 'image'
    
    Returns:
        tuple: (violation, confidence)
            violation: 1 if violent, 0 if non-violent
            confidence: float confidence score
    """
    start = time.time()
    
    if file_type == 'video':
        clips = preprocess_video(file_path)
        preds = []
        
        # Predict each 30-frame clip
        for clip in clips:
            datav = np.expand_dims(clip, axis=0)
            pred_test = model.predict(datav, verbose=0)
            fight_score = pred_test[0][1]
            preds.append(fight_score)
        
        # Average confidence across all clips
        avg_conf = float(np.mean(preds))
        
    elif file_type == 'image':
        clip = preprocess_image(file_path)
        pred_test = model.predict(clip, verbose=0)
        avg_conf = float(pred_test[0][1])
        
    else:
        raise ValueError("file_type must be 'video' or 'image'")
    
    # Determine violation (1 = violent, 0 = non-violent)
    violation = 1 if avg_conf >= ACCURACY_THRESHOLD else 0
    
    end = time.time()
    processing_time = int((end - start) * 1000)
    
    print(f"Processing time: {processing_time}ms")
    
    return violation, avg_conf

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