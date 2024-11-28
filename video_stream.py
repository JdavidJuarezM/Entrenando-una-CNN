import cv2

def generate_frames():
    cap = cv2.VideoCapture(0)  # Intenta acceder a la cámara predeterminada

    if not cap.isOpened():
        print("Error: No se puede acceder a la cámara")
        return

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            else:
                # Codifica el frame como JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        cap.release()