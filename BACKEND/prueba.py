import cv2

url = "rtsp://admin:Jhonr2005@192.168.1.30:554/Streaming/Channels/101"

cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("❌ No se pudo abrir la cámara.")
else:
    print("✅ Cámara abierta correctamente.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ No se pudo leer frame.")
        break

    cv2.imshow("RTSP Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
