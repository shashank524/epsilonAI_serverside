import cv2
from model import load_model
import numpy as np 
from scipy.misc import imresize
from test import mean_squared_loss
from keras.models import load_model
import argparse
import os
from event_recognition_api import check_for_anomalies



video=cv2.VideoCapture(0)


while video.isOpened:
	result = check_for_anomalies(video)
	print(result)

