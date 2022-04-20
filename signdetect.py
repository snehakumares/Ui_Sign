import pickle
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from predict import predict_on_video
from tensorflow.keras.models import load_model

def detect(video_path):
    IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
    SEQUENCE_LENGTH = 20
    CLASSES_LIST = ["wound", "allergy", "cough"]

    LRCN_model = load_model('LRCN.h5')
    filenm = 'LRCNModel.pickle'
    # LRCN_model = pickle.load(open("model/"+filenm, 'rb'))
    prediction = predict_on_video(video_path, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, CLASSES_LIST, LRCN_model)
    return(prediction)
    pass

