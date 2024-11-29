import cv2
import mediapipe as mp
import os

# Configuración de la carpeta para almacenar imágenes
señas = ['amor_paz', 'aceptacion', 'declinacion']
direccion = r'D:\Octavo Semestre\Vision Artificial\Actividad. Entrenando una CNN'  # Usa prefijo 'r' para cadenas sin escapes

# Crear las carpetas si no existen
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

# Contadores para las imágenes capturadas de cada seña
contadores = {seña: 0 for seña in señas}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No se puede acceder al video o fin del video.")
        break
    
    # Conversión de color
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = hands.process(color)
    
    # Dibujar las marcas de las manos detectadas
    frame.flags.writeable = True
    if resultado.multi_hand_landmarks:
        for mano in resultado.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, mano, mp_hands.HAND_CONNECTIONS)
    
    # Mostrar las opciones de captura en el frame
    cv2.putText(frame, "a: Amor y Paz | b: Aceptacion | c: Declinacion", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Mostrar el frame
    cv2.imshow("Captura de Señas", frame)
    
    # Manejo de la entrada del teclado
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
                x_max, y_max = max(x_max, x), max(y_max, y)
                x_min, y_min = min(x_min, x), min(y_min, y)

            # Verificar que los límites estén dentro del frame
            x1, y1 = max(x_min - 20, 0), max(y_min - 20, 0)
            x2, y2 = min(x_max + 20, frame.shape[1]), min(y_max + 20, frame.shape[0])

            # Recortar y guardar la imagen de la seña
            if y2 > y1 and x2 > x1:  # Verificación básica para evitar errores
                senal_img = frame[y1:y2, x1:x2]
                senal_img = cv2.resize(senal_img, (200, 200), interpolation=cv2.INTER_CUBIC)
            
                ruta_guardado = os.path.join(direccion, seña, f"{contadores[seña]}.jpg")
                cv2.imwrite(ruta_guardado, senal_img)
                contadores[seña] += 1
                print(f"Seña guardada en: {ruta_guardado}")
            
cap.release()
cv2.destroyAllWindows()