from proyecto.Validacion import procesar_frame
from flask import Flask, Response, render_template
import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np

cnn = tf.keras.models.load_model('ModeloS.keras')
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
clases = ['amor_paz', 'aceptacion', 'declinacion', 'otra_clase']

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html') 

import sys
print(sys.path)
from proyecto.Validacion import procesar_frame

#from Validacion import procesar_frame

def gen():
    cap = cv2.VideoCapture(0) 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = procesar_frame(frame) # Llamada a la función procesar_frame

        # Codificar el resultado para HTML
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/Validacion")
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='localhost', debug=True)