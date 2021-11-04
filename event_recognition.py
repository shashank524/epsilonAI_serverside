import cv2
from model import load_model
import numpy as np 
from scipy.misc import imresize
from test import mean_squared_loss
from keras.models import load_model
import argparse
from flask import request, Flask

app = Flask(__name__)

print('Loading model')
model=load_model('AnomalyDetector.h5')
print('Model loaded')
# vc=cv2.VideoCapture(0)


threshold = 0.0008
# threshold = 0.00065

def check_for_anomalies(imagedump):

    imagedump=np.array(imagedump)

    imagedump.resize(227,227,10)
    imagedump=np.expand_dims(imagedump,axis=0)
    imagedump=np.expand_dims(imagedump,axis=4)

    output=model.predict(imagedump)



    loss=mean_squared_loss(imagedump,output)


    if loss>threshold:
        result = 'Anomalies Detected'
    else:
        result = 'No Anomalies'
    
    # return result
    return result

# with app.app_context():
#     print(check_for_anomalies(threshold))
# sys.stdout.flush()

if __name__ == '__main__':
	# without SSL
    app.run(debug=True, host='0.0.0.0')
