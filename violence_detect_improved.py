# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress TensorFlow logs
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# import tensorflow as tf
# import numpy as np
# import cv2
# import time
# from skimage.transform import resize

# # === Define the model loader ===
# def mamon_videoFightModel2(weight='mamonbest947oscombo-drive.h5'):
#     layers = tf.keras.layers
#     models = tf.keras.models
#     num_classes = 2

#     base_model = tf.keras.applications.vgg19.VGG19(
#         include_top=False,
#         weights='imagenet',
#         input_shape=(160, 160, 3)
#     )

#     cnn = models.Sequential([
#         base_model,
#         layers.Flatten()
#     ])

#     model = models.Sequential([
#         layers.TimeDistributed(cnn, input_shape=(30, 160, 160, 3)),
#         layers.LSTM(30, return_sequences=True),
#         layers.TimeDistributed(layers.Dense(90)),
#         layers.Dropout(0.1),
#         layers.GlobalAveragePooling1D(),
#         layers.Dense(512, activation='relu'),
#         layers.Dropout(0.3),
#         layers.Dense(num_classes, activation='sigmoid')
#     ])

#     adam = tf.keras.optimizers.Adam(learning_rate=0.0005)
#     model.load_weights(weight)
#     model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
#     return model

# # === Function to extract frames and create clips ===
# def video_mamonreader(filename):
#     """Reads the full video and returns a list of 30-frame clips."""
#     vc = cv2.VideoCapture(filename)
    
#     if not vc.isOpened():
#         raise ValueError(f"Unable to open video file: {filename}")
    
#     all_frames = []

#     # Read ALL frames
#     while True:
#         rval, frame = vc.read()
#         if not rval:
#             break
#         frm = resize(frame, (160, 160, 3))
#         if np.max(frm) > 1:
#             frm = frm / 255.0
#         all_frames.append(frm)
#     vc.release()

#     if len(all_frames) == 0:
#         raise ValueError(f"No frames found in video: {filename}")

#     # Split into 30-frame chunks
#     clips = []
#     for i in range(0, len(all_frames), 30):
#         chunk = all_frames[i:i + 30]
#         if len(chunk) < 30:
#             # pad the last chunk if shorter than 30
#             pad_len = 30 - len(chunk)
#             chunk += [chunk[-1]] * pad_len  # Use last frame instead of zeros
#         clips.append(np.array(chunk))

#     return np.array(clips)  # shape: (num_clips, 30, 160, 160, 3)

# # === Function to process image ===
# def image_mamonreader(filename):
#     """Reads an image and creates a 30-frame clip by replicating it."""
#     frame = cv2.imread(filename)
    
#     if frame is None:
#         raise ValueError(f"Unable to load image: {filename}")
    
#     # Convert BGR to RGB
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
#     # Resize and normalize
#     frm = resize(frame, (160, 160, 3))
#     if np.max(frm) > 1:
#         frm = frm / 255.0
    
#     # Create 30 identical frames
#     frames = np.array([frm] * 30)
    
#     return np.expand_dims(frames, axis=0)  # shape: (1, 30, 160, 160, 3)

# # === Prediction wrapper ===
# def pred_fight(model, video, accuracy_threshold=0.65):
#     pred_test = model.predict(video, verbose=0)
#     fight_score = pred_test[0][1]
#     return (fight_score >= accuracy_threshold, fight_score)

# # === Run on video ===
# def main_fight(video_path, accuracy_threshold=0.65):
#     clips = video_mamonreader(video_path)
#     model = mamon_videoFightModel2(weight='mamonbest947oscombo-drive.h5')

#     start = time.time()
#     preds = []

#     # Predict each 30-frame clip
#     for clip in clips:
#         datav = np.expand_dims(clip, axis=0)
#         _, conf = pred_fight(model, datav, accuracy_threshold=accuracy_threshold)
#         preds.append(conf)

#     end = time.time()

#     # Combine results from all clips
#     avg_conf = float(np.mean(preds))
#     is_fight = avg_conf >= accuracy_threshold

#     result = {
#         'fight': is_fight,
#         'confidence': avg_conf,
#         'num_segments': len(preds),
#         'processing_time_ms': int((end - start) * 1000)
#     }
#     return result

# # === Run on image ===
# def main_fight_image(image_path, accuracy_threshold=0.65):
#     clip = image_mamonreader(image_path)
#     model = mamon_videoFightModel2(weight='mamonbest947oscombo-drive.h5')

#     start = time.time()
#     _, conf = pred_fight(model, clip, accuracy_threshold=accuracy_threshold)
#     end = time.time()

#     is_fight = conf >= accuracy_threshold

#     result = {
#         'fight': is_fight,
#         'confidence': float(conf),
#         'num_segments': 1,
#         'processing_time_ms': int((end - start) * 1000)
#     }
#     return result

# # === Unified prediction function ===
# def predict_violation(input_path, input_type='video', accuracy_threshold=0.65):
#     """
#     Unified function to predict violence in video or image.
    
#     Args:
#         input_path: Path to video or image file
#         input_type: 'video' or 'image'
#         accuracy_threshold: Confidence threshold for classification
    
#     Returns:
#         Dictionary with prediction results
#     """
#     if input_type == 'video':
#         result = main_fight(input_path, accuracy_threshold)
#     elif input_type == 'image':
#         result = main_fight_image(input_path, accuracy_threshold)
#     else:
#         raise ValueError("input_type must be 'video' or 'image'")
    
#     label = "violent" if result['fight'] else "non-violent"
#     print(f"Prediction: {label} (confidence: {result['confidence']:.4f})")
#     print(f"Processing time: {result['processing_time_ms']}ms")
#     print(f"Number of segments analyzed: {result['num_segments']}")
    
#     return result

# # === Example usage ===
# if __name__ == "__main__":
#     video_path = "Violence/v_test.mp4"
#     image_path = "uploads/posts/images/v1_test.jpg"

#     # Predict video
#     print("=== Video Prediction ===")
#     video_result = predict_violation(video_path, input_type='video', accuracy_threshold=0.65)
#     print()

#     # # Predict image
#     # print("=== Image Prediction ===")
#     # image_result = predict_violation(image_path, input_type='image', accuracy_threshold=0.65)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import numpy as np
import cv2
import time
from skimage.transform import resize

# === Define the model loader ===
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

# ✅ FIX 1: Load model ONCE at module level (not in functions!)
print("Loading model... (this may take a minute)")
MODEL = mamon_videoFightModel2(weight='mamonbest947oscombo-drive.h5')
print("Model loaded successfully!")

# === Function to extract frames and create clips ===
def video_mamonreader(filename):
    """Reads the full video and returns a list of 30-frame clips."""
    vc = cv2.VideoCapture(filename)
    
    if not vc.isOpened():
        raise ValueError(f"Unable to open video file: {filename}")
    
    all_frames = []

    # Read ALL frames
    while True:
        rval, frame = vc.read()
        if not rval:
            break
        # ✅ FIX 2: Convert BGR to RGB before resizing!
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frm = resize(frame, (160, 160, 3))
        if np.max(frm) > 1:
            frm = frm / 255.0
        all_frames.append(frm)
    vc.release()

    if len(all_frames) == 0:
        raise ValueError(f"No frames found in video: {filename}")

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

# === Function to process image ===
def image_mamonreader(filename):
    """Reads an image and creates a 30-frame clip by replicating it."""
    frame = cv2.imread(filename)
    
    if frame is None:
        raise ValueError(f"Unable to load image: {filename}")
    
    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize and normalize
    frm = resize(frame, (160, 160, 3))
    if np.max(frm) > 1:
        frm = frm / 255.0
    
    # Create 30 identical frames
    frames = np.array([frm] * 30)
    
    return np.expand_dims(frames, axis=0)  # shape: (1, 30, 160, 160, 3)

# === Prediction wrapper ===
def pred_fight(video, accuracy_threshold=0.65):
    """Use the global MODEL instead of passing it"""
    pred_test = MODEL.predict(video, verbose=0)
    fight_score = pred_test[0][1]
    return (fight_score >= accuracy_threshold, fight_score)

# === Run on video ===
def main_fight(video_path, accuracy_threshold=0.65):
    clips = video_mamonreader(video_path)
    # ✅ FIX 3: Removed model loading from here - use global MODEL

    start = time.time()
    preds = []

    # Predict each 30-frame clip
    for clip in clips:
        datav = np.expand_dims(clip, axis=0)
        _, conf = pred_fight(datav, accuracy_threshold=accuracy_threshold)
        preds.append(conf)

    end = time.time()

    # Combine results from all clips
    avg_conf = float(np.mean(preds))
    is_fight = avg_conf >= accuracy_threshold

    result = {
        'fight': is_fight,
        'confidence': avg_conf,
        'num_segments': len(preds),
        'processing_time_ms': int((end - start) * 1000)
    }
    return result

# === Run on image ===
def main_fight_image(image_path, accuracy_threshold=0.65):
    clip = image_mamonreader(image_path)
    # ✅ FIX 4: Removed model loading from here - use global MODEL

    start = time.time()
    _, conf = pred_fight(clip, accuracy_threshold=accuracy_threshold)
    end = time.time()

    is_fight = conf >= accuracy_threshold

    result = {
        'fight': is_fight,
        'confidence': float(conf),
        'num_segments': 1,
        'processing_time_ms': int((end - start) * 1000)
    }
    return result

# === Unified prediction function ===
def predict_violation(input_path, input_type='video', accuracy_threshold=0.65):
    """
    Unified function to predict violence in video or image.
    
    Args:
        input_path: Path to video or image file
        input_type: 'video' or 'image'
        accuracy_threshold: Confidence threshold for classification
    
    Returns:
        Dictionary with prediction results
    """
    if input_type == 'video':
        result = main_fight(input_path, accuracy_threshold)
    elif input_type == 'image':
        result = main_fight_image(input_path, accuracy_threshold)
    else:
        raise ValueError("input_type must be 'video' or 'image'")
    
    label = "violent" if result['fight'] else "non-violent"
    print(f"Prediction: {label} (confidence: {result['confidence']:.4f})")
    print(f"Processing time: {result['processing_time_ms']}ms")
    print(f"Number of segments analyzed: {result['num_segments']}")
    
    return result

# === Example usage ===
if __name__ == "__main__":
    video_path = "Violence/nv4_test.mp4"
    image_path = "uploads/posts/images/v1_test.jpg"

    # Predict video
    print("\n=== Video Prediction ===")
    video_result = predict_violation(video_path, input_type='video', accuracy_threshold=0.65)
    print()

    # # Predict image
    # print("=== Image Prediction ===")
    # image_result = predict_violation(image_path, input_type='image', accuracy_threshold=0.65)