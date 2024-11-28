import cv2

def generate_frames():
    cap = cv2.VideoCapture(1)  # Accede a la cámara
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
    cap.release()