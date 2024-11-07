import cv2
import mediapipe as mp
import os

# Configuración de la carpeta para almacenar imágenes
señas = ['amor_paz', 'aceptacion', 'declinacion']
direccion = 'D:\Octavo Semestre\Vision Artificial\Actividad. Entrenando una CNN'

for seña in señas:
    carpeta = os.path.join(direccion, seña)
    if not os.path.exists(carpeta):
        print(f'Carpeta creada: {carpeta}')
        os.makedirs(carpeta)

# Inicializamos la captura de video
cap = cv2.VideoCapture(0)

# Configuración de MediaPipe para la detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

contadores = {seña: 0 for seña in señas}  # Contadores para las imágenes capturadas de cada seña

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = hands.process(color)
    
    if resultado.multi_hand_landmarks:
        for mano in resultado.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, mano, mp_hands.HAND_CONNECTIONS)
    
    # Mostrar las opciones de captura en el frame
    cv2.putText(frame, "a: Amor y Paz | b: Aceptacion | c: Declinacion", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Captura de Señas", frame)
    
    key = cv2.waitKey(1)
    if key == 27:  # ESC para salir
        break
    elif key == ord('a'):
        seña = 'amor_paz'
    elif key == ord('b'):
        seña = 'aceptacion'
    elif key == ord('c'):
        seña = 'declinacion'
    else:
        continue
    
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
            senal_img = frame[y1:y2, x1:x2]
            senal_img = cv2.resize(senal_img, (200, 200), interpolation=cv2.INTER_CUBIC)
            
            ruta_guardado = os.path.join(direccion, seña, f"{contadores[seña]}.jpg")
            contadores[seña] += 1
            cv2.imwrite(ruta_guardado, senal_img)
            print(f"Seña guardada en: {ruta_guardado}")
            
cap.release()
cv2.destroyAllWindows()