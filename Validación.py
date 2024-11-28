import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

modelo = 'ModeloS.keras'  # Nombre del archivo de modelo actualizado
cnn = load_model(modelo)

# Mapping de clases, asegurando el mismo tamaño
clases = ['amor_paz', 'aceptacion', 'declinacion', 'otra_clase']

# Iniciar captura de video
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

try:
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
                    x_max, y_max = max(x, x_max), max(y, y_max)
                    x_min, y_min = min(x, x_min), min(y, y_min)

                x1, y1 = max(0, x_min - 20), max(0, y_min - 20)
                x2, y2 = min(frame.shape[1], x_max + 20), min(frame.shape[0], y_max + 20)
                
                señal_img = frame[y1:y2, x1:x2]
                señal_img = cv2.resize(señal_img, (200, 200), interpolation=cv2.INTER_CUBIC)
                
                x = img_to_array(señal_img) / 255.0
                x = np.expand_dims(x, axis=0)
                prediccion = cnn.predict(x)
                respuesta = np.argmax(prediccion)
                
                if 0 <= respuesta < len(clases):
                    texto_clase = clases[respuesta]
                else:
                    texto_clase = "Desconocido"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, texto_clase, (x1, y1 - 5), 1, 1.3, (0, 255, 0), 1, cv2.LINE_AA)
        
        cv2.imshow("Video", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()