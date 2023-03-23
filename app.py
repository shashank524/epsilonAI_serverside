import os
import cv2
import numpy as np 
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from event_recognition import check_for_anomalies
from flask import Flask, request, Response, jsonify

app = Flask(__name__)
sess = tf.compat.v1.keras.backend.get_session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)
graph = tf.compat.v1.get_default_graph()


# for CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST') # Put any other methods you need here
    return response


@app.route('/', methods=['POST'])
def image():
    try:
        imagedump = []
        source = request.form.get("source")
        print(f"Processing video from source: {source}")
        video_stream = cv2.VideoCapture(source)

        num_frames = 5
        for i in range(num_frames):
            rval, frame = video_stream.read()
            if not rval:
                break
            frame = cv2.resize(frame, (227, 227))
            gray = 0.2989 * frame[:, :, 0] + 0.5870 * frame[:, :, 1] + 0.1140 * frame[:, :, 2]
            gray = (gray - gray.mean()) / gray.std()
            gray = np.clip(gray, 0, 1)
            imagedump.append(gray)

        global graph
        global sess
        with graph.as_default():
            set_session(sess)
            result = check_for_anomalies(imagedump)

        print(f"Result: {result}")
        return jsonify({"result": result, "camera": source})

    except Exception as e:
        print('POST /image error: %s' % str(e))
        return str(e)


if __name__ == '__main__':
    # without SSL
    app.run(debug=False, port=5000)

    # with SSL
    # app.run(debug=True, host='0.0.0.0', ssl_context=('ssl/server.crt', 'ssl/server.key'))
