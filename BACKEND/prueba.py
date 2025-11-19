# prueba.py
import os
import time
import logging
from datetime import datetime
from tkinter import Tk, filedialog
import socketio
from services.video_service import VideoService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PRUEBA_VIDEO_GUI")

# =====================================================
# 🔌 CONEXIÓN SOCKET (AJUSTADO + LOGS DE DEPURACIÓN)
# =====================================================
sio = socketio.Client(logger=True, engineio_logger=True)

SOCKET_URL = "http://127.0.0.1:5000"   # ⚠️ Usa tu backend local
# SOCKET_URL = "https://accident-detector.site"  # Úsalo solo si ese dominio está activo

def conectar_socket():
    global sio
    try:
        sio.connect(
            SOCKET_URL,
            transports=['websocket'],
            namespaces=['/']
        )
        logger.info("✅ Conectado correctamente a Socket.IO")
        return True
    except Exception as e:
        logger.error(f"❌ Error al conectar con Socket.IO: {e}")
        return False


# =====================================================
# 🖼️ SELECCIONAR VIDEO
# =====================================================
def seleccionar_video():
    root = Tk()
    root.withdraw()
    archivo = filedialog.askopenfilename(
        title="Seleccionar video",
        filetypes=[("MP4 files", "*.mp4"), ("Todos los archivos", "*.*")]
    )

    if archivo:
        logger.info(f"🎥 Video seleccionado: {archivo}")
        analizar_video(archivo)
    else:
        logger.info("⚠️ No se seleccionó ningún video.")


# =====================================================
# 🤖 ANALIZAR VIDEO + ENVIAR ALERTA
# =====================================================
def analizar_video(video_path):
    logger.info("\n" + "=" * 70)
    logger.info("🎮 INICIANDO ANÁLISIS DE VIDEO")
    logger.info("=" * 70 + "\n")

    # Ejecutar detección con tu servicio
    result = VideoService.analyze_video(video_path, os.path.basename(video_path))

    total = result.get('total_detections', 0)
    logger.info(f"📊 Total detecciones: {total}")

    # Si no hubo detecciones → no se envía alerta
    if total == 0:
        logger.info("⚠️ Sin accidentes detectados, no se envía alerta.")
        return

    # -------------------------------------
    # ✨ PREPARAR LA ALERTA A ENVIAR
    # -------------------------------------
    first_det = result['detections'][0]

    accident_id = int(time.time())  # simular ID único
    camera_id = "CAM_PRUEBA_001"
    camera_ip = "127.0.0.1"

    latitude = 4.710989
    longitude = -74.072090
    confidence = int(first_det['confidence'] * 100)

    payload = {
        "accident_id": accident_id,
        "camera_id": camera_id,
        "camera_ip": camera_ip,
        "latitude": latitude,
        "longitude": longitude,
        "timestamp": datetime.now().isoformat(),
        "image_url": f"/api/mobile/image/{accident_id}",  # coincide con tu backend
        "message": f"🚨 Accidente detectado en cámara {camera_id}",
        "severity": "high",
        "confidence": confidence,
    }

    # -------------------------------------
    # 🚨 ENVIAR ALERTA POR SOCKET.IO
    # -------------------------------------
    if sio.connected:
        try:
            sio.emit("mobile_emergency_alert", payload)
            logger.info("🔺 ALERTA ENVIADA EXITOSAMENTE")
            logger.info(payload)
        except Exception as e:
            logger.error(f"❌ Error enviando alerta: {e}")
    else:
        logger.error("❌ No hay conexión con Socket.IO — alerta NO enviada")

    # Exportar reporte (si aplica)
    VideoService.generate_report(result)
    logger.info("✅ Análisis completado.\n")


# =====================================================
# 🚀 EJECUCIÓN PRINCIPAL
# =====================================================
if __name__ == "__main__":
    if conectar_socket():
        seleccionar_video()
        sio.disconnect()
        logger.info("🔌 SocketIO desconectado")
    else:
        logger.error("❌ No se pudo iniciar prueba — Socket no conectado")
