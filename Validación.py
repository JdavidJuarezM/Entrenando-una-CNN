import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

modelo = 'ModeloS.keras'  # Nombre del archivo de modelo actualizado
peso =  'pesosS.weights.h5'  # Nombre del archivo de pesos actualizado
cnn = load_model(modelo)  # Cargamos el modelo
cnn.load_weights(peso)  # Cargamos los pesos

# Mapping de clases
clases = ['amor_paz', 'aceptacion', 'declinacion']

# Iniciar captura de video
cap = cv2.VideoCapture(0)

# Configuración de MediaPipe para la detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = hands.process(color)
    
    if resultado.multi_hand_landmarks:
        for mano in resultado.multi_hand_landmarks:
            x_max, y_max = (0, 0)
            x_min, y_min = (frame.shape[1], frame.shape[0])
            for lm in mano.landmark:
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                if x > x_max: x_max = x
                if x < x_min: x_min = x
                if y > y_max: y_max = y
                if y < y_min: y_min = y

            x1, y1 = x_min - 20, y_min - 20
            x2, y2 = x_max + 20, y_max + 20
            señal_img = frame[y1:y2, x1:x2]
            señal_img = cv2.resize(señal_img, (200, 200), interpolation=cv2.INTER_CUBIC)
            
            x = img_to_array(señal_img) / 255.0  # Normalizar la imagen tal como en el entrenamiento
            x = np.expand_dims(x, axis=0)
            resultado = cnn.predict(x)
            respuesta = np.argmax(resultado)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, '{}'.format(clases[respuesta]), (x1, y1 - 5), 1, 1.3, (0, 255, 0), 1, cv2.LINE_AA)
    
    cv2.imshow("Video", frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()