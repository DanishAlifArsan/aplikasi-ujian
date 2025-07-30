import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from keras.models import load_model
from scipy.spatial import distance
from imageio import imread
from skimage.transform import resize
import sys
sys.path.append('../code/')
import inception_resnet_v1 as Inception

mediapipe_path = '../model/mediapipe/blaze_face_short_range.tflite'
model_path = '/workspaces/aplikasi-ujian/model/facenet_2/model/skenario_2/facenet_keras.keras'
model = load_model(model_path, compile=False, custom_objects={'Custom>scaling': Inception.scaling, 'l2_norm': Inception.l2_norm})

def load_and_align_images(image, margin):
    BaseOptions = mp.tasks.BaseOptions
    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    # Create a face detector instance with the image mode:
    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=mediapipe_path),
        running_mode=VisionRunningMode.IMAGE)
    with FaceDetector.create_from_options(options) as detector:
      # The detector is initialized. Use it here.
      # ...
        img = imread(image)
        mp_image = mp.Image.create_from_file(image)
        faces = detector.detect(mp_image)
        if len(faces.detections) > 0 :
            bbox = faces.detections[0].bounding_box
            x = bbox.origin_x
            y = bbox.origin_y
            w = bbox.width
            h = bbox.height
            cropped = img[y-margin//2:y+h+margin//2,
                      x-margin//2:x+w+margin//2, :]
            aligned_images = resize(cropped, (160, 160), mode='reflect')
            
            return aligned_images, x, y, w, h

def load_and_align_videos(red,frame,cap):
    BaseOptions = mp.tasks.BaseOptions
    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create a face detector instance with the video mode:
    options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=mediapipe_path),
    running_mode=VisionRunningMode.VIDEO)
    with FaceDetector.create_from_options(options) as detector:
    # The detector is initialized. Use it here.
    # ...
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        frame_timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        
        faces = detector.detect_for_video(mp_image, int(frame_timestamp_ms * 1000))

        if faces.detections:
            # Ambil bounding box wajah
            bboxC = faces.detections[0].bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.origin_x), int(bboxC.origin_y), int(bboxC.width), int(bboxC.height)

            # Potong wajah dari gambar
            margin = 10
            face_image = frame[y-margin//2:y+h+margin//2, x-margin//2:x+w+margin//2, :]
            if face_image is not None and face_image.size > 0:
                face_image = resize(face_image, (160, 160), mode='reflect')
    
    return face_image

def get_embedding(face):
    sample = np.expand_dims(face, axis=0)
    return model.predict(sample)

def calc_dist(embedding1, embedding2):
    return np.linalg.norm(embedding1-embedding2)

def get_data():
    database = {}
    for filename in os.listdir('../database'): 
        path = os.path.join('../database/', filename)
        face = load_and_align_images(path, 10)
        if face is not None:
            emb = get_embedding(face)
            name, _ = os.path.splitext(filename)
            database[name] = emb
    
    return database