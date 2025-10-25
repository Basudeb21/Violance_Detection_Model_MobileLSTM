# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Optional: disable oneDNN optimizations

# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model


# # Load the trained model
# model = load_model("model/MobBiLSTM_model_saved101.keras")

# # Video path
# video_path = "uploads/posts/videos/nv2_test.mp4"

# # Parameters
# FRAME_SIZE = (64, 64)
# FRAMES_PER_CLIP = 16

# def preprocess_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     frames = []
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame = cv2.resize(frame, FRAME_SIZE)
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # match training preprocessing
#         frames.append(frame)
#     cap.release()

#     # If video has fewer frames than required, pad with last frame
#     if len(frames) < FRAMES_PER_CLIP:
#         frames.extend([frames[-1]] * (FRAMES_PER_CLIP - len(frames)))
#     else:
#         frames = frames[:FRAMES_PER_CLIP]

#     # Normalize
#     frames = np.array(frames) / 255.0
#     # Add batch dimension: (1, 15, 128, 128, 3)
#     return np.expand_dims(frames, axis=0)

# # Preprocess the video
# video_clip = preprocess_video(video_path)

# # Predict
# prediction = model.predict(video_clip)[0][0]
# print("prediction",prediction)
# # Output
# if prediction >= 0.5:
#     print(f"Prediction: non-violent ({prediction:.4f})")
# else:
#     print(f"Prediction: Violent ({prediction:.4f})")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("model/MobBiLSTM_model_saved101.keras")

# Parameters
FRAME_SIZE = (64, 64)
FRAMES_PER_CLIP = 16

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
    # Read image
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Image not found: {image_path}")
    frame = cv2.resize(frame, FRAME_SIZE)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Replicate image to create a fake video clip
    frames = np.array([frame] * FRAMES_PER_CLIP) / 255.0
    return np.expand_dims(frames, axis=0)

def predict_violation(input_path, input_type='video'):
    if input_type == 'video':
        clip = preprocess_video(input_path)
    elif input_type == 'image':
        clip = preprocess_image(input_path)
    else:
        raise ValueError("input_type must be 'video' or 'image'")

    prediction = model.predict(clip)[0][0]

    if prediction >= 0.5:
        label = "non-violent"
    else:
        label = "violent"
    
    print(f"Prediction: {label} ({prediction:.4f})")
    return prediction

# Example usage:
video_path = "uploads/posts/videos/nv2_test.mp4"
image_path = "uploads/posts/images/nv2_test.jpg"

# Predict video
# predict_violation(video_path, input_type='video')

# Predict image
predict_violation(image_path, input_type='image')
