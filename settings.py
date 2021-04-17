import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_IMAGES_FOLDER = os.path.join(BASE_DIR, 'data', 'images')
MODEL_DIR = os.path.join(BASE_DIR, 'data', 'model')
FONTS_DIR = os.path.join(BASE_DIR, 'data', 'fonts')
IMAGE_AMOUNT_FOR_TRAIN = 100
CAMERA_SOURCE = 0