import os
import sys


MODEL_NAME = "plan_classifier"
NUMBER_EPOCHS = 20
NUMBER_CLASSES = 2
BATCH_SIZE = 50
LEARNING_RATE = 0.001
CROP_SIZE = 400
READ_HISTORY_MODEL = True
# Input size dimension
INPUT_DIMENSION = 3
CLASS_LABELS = ('Other', 'Plans')


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_IMAGES_PATH = os.path.join(BASE_DIR, 'images', 'root_data')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'models', MODEL_NAME)
