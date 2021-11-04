import cv2
from model import load_model
import numpy as np 
from scipy.misc import imresize
from test import mean_squared_loss
from keras.models import load_model
import argparse
import os
from event_recognition_api import check_for_anomalies



VIDEO_NAME = 'Abuse.mp4'
CWD_PATH = os.getcwd()
PATH_TO_VIDEO = os.path.join(CWD_PATH, VIDEO_NAME)

video=cv2.VideoCapture(PATH_TO_VIDEO)


while video.isOpened:
	result = check_for_anomalies(video)
	print(result)