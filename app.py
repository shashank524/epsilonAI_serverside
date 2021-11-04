import os
from test import mean_squared_loss
from model import load_model
from PIL import Image
import cv2
from scipy.misc import imresize
import numpy as np 
import keras
from event_recognition import check_for_anomalies
from flask import Flask, request, Response, jsonify, render_template
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
# import pafy
# import youtube_dl


app = Flask(__name__)
# app = Flask(__name__, template_folder='./static/')
sess = keras.backend.get_session()
init = tf.global_variables_initializer()
sess.run(init)
graph = tf.get_default_graph()



# for CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST') # Put any other methods you need here
    return response

'''
@app.route('/')
def index():
    # return Response('event recognition demo')
    return render_template('home.html')


@app.route('/video')
def remote():
    return Response(open('./static/video.html').read(), mimetype="text/html")

'''
@app.route('/', methods=['POST'])

def image():
    try:
        # Set an image confidence threshold value to limit returned data
        # threshold = 0.0008
        imagedump=[]
        source = request.form.get("source")
        print(source)
        video_stream=cv2.VideoCapture(source)
        
        # vPafy = pafy.new(source)
        # play = vPafy.getbest(preftype="mp4")
        # video_stream=cv2.VideoCapture(play.url)

        for i in range(10):
            rval,frame=video_stream.read()
            frame=imresize(frame,(227,227,3))

            #Convert the Image to Grayscale

            gray=0.2989*frame[:,:,0]+0.5870*frame[:,:,1]+0.1140*frame[:,:,2]
            gray=(gray-gray.mean())/gray.std()
            gray=np.clip(gray,0,1)
            imagedump.append(gray)
            global graph
            global sess
            with graph.as_default():
                set_session(sess)
                result = check_for_anomalies(imagedump)
        print(result)
        return jsonify({"result" : result, "camera": source})

    except Exception as e:
        print('POST /image error: %e' % e)
        return e

if __name__ == '__main__':
	# without SSL
    app.run(debug=False, port=5000)

	# with SSL
    #app.run(debug=True, host='0.0.0.0', ssl_context=('ssl/server.crt', 'ssl/server.key'))
