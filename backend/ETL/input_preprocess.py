import numpy as np
from skimage.io import imread
import io

def cut_image(img, low=0.2, high=0.8):
    h, w = img.shape[:2]
    return img[int(low*h):int(high*h), int(low*w):int(high*w), :]

def calculate_features(img, percentiles=[5,15,25,50,75,85,95]):
    img_cut = cut_image(img)
    features = []
    for i in range(3):  # R, G, B
        for p in percentiles:
            features.append(np.percentile(img_cut[:, :, i], p))
    return np.array(features)

def normalize_features(features, image):
    white_patch = image[300:350, 300:350]
    white_ref = [np.median(white_patch[:, :, i]) for i in range(3)]
    normalized = features.copy()
    for i in range(len(features)):
        color_idx = i // 7
        white_val = white_ref[color_idx] or 1
        normalized[i] /= white_val
    return normalized

def extract_features_from_image(image_bytes):
    image = imread(io.BytesIO(image_bytes))
    features = calculate_features(image)
    features = normalize_features(features, image)
    return features