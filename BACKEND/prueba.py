from flask import Flask, Response
import cv2

app = Flask(__name__)

# Tu RTSP
RTSP_URL = "rtsp://admin:Jhonr2005@192.168.1.30:554/Streaming/Channels/101"

def generate_frames():
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        # Comprimir frame a JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        # Devuelve un frame para streaming MJPEG
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """
    <html>
    <body>
        <h1>Test RTSP</h1>
        <img src="/video" width="640" height="480">
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
