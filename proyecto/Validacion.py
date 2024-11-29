import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

def procesar_frame(frame):
    # Cargar el modelo de TensorFlow
    modelo = 'ModeloS.keras'  # Nombre del archivo de modelo
    cnn = load_model(modelo)

    # Mapeo de clases, asegurando el mismo tamaño
    clases = ['amor_paz', 'aceptacion', 'declinacion', 'otra_clase']

    # Configuración de MediaPipe para detección de manos
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Convertir el frame de BGR a RGB
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = hands.process(color)

    # Dibujar marcas de la mano detectada
    frame.flags.writeable = True
    if resultado.multi_hand_landmarks:
        for mano in resultado.multi_hand_landmarks:
            x_max, y_max = (0, 0)
            x_min, y_min = (frame.shape[1], frame.shape[0])

            # Encontrar los límites de la mano
            for lm in mano.landmark:
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                x_max, y_max = max(x_max, x), max(y_max, y)
                x_min, y_min = min(x_min, x), min(y_min, y)

            # Ajustar límites y recortar la imagen de la mano
            x1, y1 = max(0, x_min - 20), max(0, y_min - 20)
            x2, y2 = min(frame.shape[1], x_max + 20), min(frame.shape[0], y_max + 20)

            # Verificar que los límites no estén invertidos
            if y2 > y1 and x2 > x1:
                señal_img = frame[y1:y2, x1:x2]

                # Preparar la imagen para la predicción si el recorte es válido
                if señal_img.size > 0:
                    señal_img = cv2.resize(señal_img, (200, 200), interpolation=cv2.INTER_CUBIC)
                    x = img_to_array(señal_img) / 255.0
                    x = np.expand_dims(x, axis=0)

                    # Realizar la predicción
                    prediccion = cnn.predict(x)
                    respuesta = np.argmax(prediccion)

                    # Asignar texto de la clase predicha
                    texto_clase = clases[respuesta] if 0 <= respuesta < len(clases) else "Desconocido"

                    # Dibujar rectángulo y texto en el frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame, texto_clase, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    return frame