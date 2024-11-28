from flask import Flask, Response
from video_stream import generate_frames  # Importa la lógica de transmisión

app = Flask(_name_)

@app.route('/video_feed')
def video_feed():
    # Endpoint para el feed de video
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    # Página principal
    return '''
    <html>
        <head>
            <title>Stream de Cámara</title>
        </head>
        <body>
            <h1>Video en Vivo</h1>
            <img src="/video_feed" alt="Video Stream">
        </body>
    </html>
    '''

if _name_ == '_main_':
    app.run(debug=True)