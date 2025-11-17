import cv2
import threading
import time
import logging
import base64
from datetime import datetime
from config import Config
from database import db
import numpy as np
from collections import deque
import os
from urllib.parse import quote

# --- Cifrado y descifrado seguro con Fernet ---
try:
    from cryptography.fernet import Fernet, InvalidToken
except Exception:
    Fernet = None
    InvalidToken = Exception

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== CONFIGURAR CIFRADO ====================
FERNET_INSTANCE = None

def _get_fernet():
    """Obtiene una instancia Fernet desde Config (si existe clave válida)."""
    key = getattr(Config, "FERNET_KEY", None) or getattr(Config, "ENCRYPTION_KEY", None)
    if not key or Fernet is None:
        return None
    try:
        if isinstance(key, str):
            key = key.encode()
        return Fernet(key)
    except Exception as e:
        logger.error(f"❌ Error inicializando Fernet: {e}")
        return None

FERNET_INSTANCE = _get_fernet()

def encrypt_value(value: str) -> str:
    if not value or not FERNET_INSTANCE:
        return value
    try:
        return FERNET_INSTANCE.encrypt(value.encode()).decode()
    except Exception as e:
        logger.error(f"Error cifrando valor: {e}")
        return value

def decrypt_value(value: str):
    if not value or not FERNET_INSTANCE:
        return value
    try:
        return FERNET_INSTANCE.decrypt(value.encode()).decode()
    except InvalidToken:
        return value
    except Exception:
        return value

# ============================================================

_detector_instance = None
def get_detector():
    global _detector_instance
    if _detector_instance is None:
        from models.detector import SevereAccidentDetector
        _detector_instance = SevereAccidentDetector()
    return _detector_instance


class CameraStream:
    """Streaming RTSP con cifrado de credenciales y grabación de evidencia"""
    def __init__(self, camera_data, socketio, use_yolo=True):
        self.camera_id = camera_data["id"]
        self.camera_data = camera_data
        self.socketio = socketio
        self.use_yolo = use_yolo

        self.rtsp_url = self._build_rtsp_url()
        self.is_running = False
        self.thread = None
        self.cap = None
        self.current_frame = None
        self.frame_count = 0

        # Control de FPS
        self.last_emit_time = 0
        self.emit_interval = 0.033

        # Detector
        self.detector = None

        # Control de detecciones (VALORES DINÁMICOS DESDE BD)
        self.consecutive_detections = 0
        self.required_consecutive = db.get_config('frames_requeridos') or 120
        self.last_confirmed_time = 0
        self.cooldown_seconds = db.get_config('cooldown_segundos') or 60

        # Estadísticas
        self.total_detections = 0
        self.confirmed_accidents = 0

        # Buffer de video
        self.frame_buffer = deque(maxlen=375)
        self.is_recording = False
        self.frames_to_record_after = 0
        self.recording_frames = []
        self.recording_start_time = None

        self.last_detection_bbox = None
        self.last_detection_confidence = 0

        logger.info(f"🎥 CameraStream inicializado - ID: {self.camera_id}, URL: {self.rtsp_url}")

    def _build_rtsp_url(self):
        """
        Construye y/o descifra la URL RTSP de forma segura.
        - Si viene cifrada, la descifra.
        - Si no, usa usuario y contraseña (también descifrados).
        """
        ip = self.camera_data.get("ip")
        puerto = self.camera_data.get("puerto", "554")

        # 1️⃣ Desencriptar campos (si existen)
        usuario = decrypt_value(self.camera_data.get("usuario_cifrado")) or self.camera_data.get("usuario")
        password = decrypt_value(self.camera_data.get("password_cifrado")) or self.camera_data.get("password")
        url_cifrada = decrypt_value(self.camera_data.get("url_rtsp_cifrada")) if self.camera_data.get("url_rtsp_cifrada") else None

        # 2️⃣ Si hay URL cifrada, úsala directamente
        if url_cifrada:
            return url_cifrada

        # 3️⃣ Construir URL RTSP de forma segura (sin guardar en BD)
        if usuario and password:
            usuario_encoded = quote(usuario)
            password_encoded = quote(password)
            return f"rtsp://{usuario_encoded}:{password_encoded}@{ip}:{puerto}/Streaming/Channels/101"
        else:
            return f"rtsp://{ip}:{puerto}/stream"

    # ==================== CONTROL DE STREAM ====================

    def start(self):
        if self.is_running:
            logger.warning(f"⚠️ Stream {self.camera_id} ya está corriendo")
            return False
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info(f"✅ Stream iniciado - Cámara ID: {self.camera_id}")
        return True

    def stop(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info(f"🛑 Stream detenido - Cámara ID: {self.camera_id}")

    def _capture_loop(self):
        reconnect_attempts = 0
        ffmpeg_opts = (
            " -hwaccel auto "
            " -rtsp_transport tcp "
            " -fflags nobuffer "
            " -flags low_delay "
            " -probesize 32 "
            " -analyzeduration 0 "
            " -rw_timeout 5000000 "
        )

        while self.is_running:
            try:
                logger.info(f"🔌 Intentando conectar cámara {self.camera_id} (intento {reconnect_attempts+1})...")
                self.cap = cv2.VideoCapture(self.rtsp_url + ffmpeg_opts, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                if not self.cap.isOpened():
                    reconnect_attempts += 1
                    logger.warning(f"❌ No se pudo abrir cámara {self.camera_id}. Reintentando ({reconnect_attempts})...")
                    time.sleep(3)
                    continue

                logger.info(f"✅ Cámara {self.camera_id} conectada correctamente")
                reconnect_attempts = 0

                while self.is_running and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret or frame is None:
                        logger.warning(f"⚠️ Frame perdido - Cámara {self.camera_id}")
                        time.sleep(0.05)
                        continue

                    self.frame_count += 1

                    if self.use_yolo:
                        if self.detector is None:
                            self.detector = get_detector()
                        tiene_severe, confidence, annotated, bbox = self.detector.detect_severe(frame)
                        self.current_frame = annotated
                        self.frame_buffer.append(annotated.copy())

                        if tiene_severe:
                            self.last_detection_bbox = bbox
                            self.last_detection_confidence = confidence
                            self._verify_detection(confidence, bbox)
                        else:
                            self.consecutive_detections = 0
                    else:
                        self.current_frame = frame
                        self.frame_buffer.append(frame.copy())

                    if self.is_recording:
                        self.recording_frames.append(self.current_frame.copy())
                        self.frames_to_record_after -= 1
                        if self.frames_to_record_after <= 0:
                            self._save_recording()

                    now = time.time()
                    if now - self.last_emit_time >= self.emit_interval:
                        self._emit_frame()
                        self.last_emit_time = now

            except Exception as e:
                logger.error(f"💥 Error en cámara {self.camera_id}: {e}")
            finally:
                if self.cap:
                    self.cap.release()
                    self.cap = None
                if self.is_running:
                    logger.warning(f"♻️ Intentando reconectar cámara {self.camera_id} en 2s...")
                    time.sleep(2)

    # ==================== DETECCIÓN Y REGISTRO ====================

    def _verify_detection(self, confidence, bbox):
        current_time = time.time()
        self.consecutive_detections += 1
        self.total_detections += 1

        progress_percentage = (self.consecutive_detections / self.required_consecutive) * 100
        self.socketio.emit("tentative_detection", {
            "camera_id": self.camera_id,
            "camera_ip": self.camera_data.get("ip", "Unknown"),
            "confidence": round(confidence * 100, 2),
            "consecutive_frames": self.consecutive_detections,    # ✅ AGREGADO
            "required_frames": self.required_consecutive,         # ✅ AGREGADO
            "progress": min(progress_percentage, 100),
            "timestamp": datetime.now().isoformat(),
            "status": "VERIFICANDO",
            "bbox": bbox
        })

        if (self.consecutive_detections >= self.required_consecutive and
            current_time - self.last_confirmed_time >= self.cooldown_seconds):
            self.confirmed_accidents += 1
            self.last_confirmed_time = current_time
            self.consecutive_detections = 0
            self._start_recording()
            self._emit_confirmed(confidence, bbox)
            logger.warning(f"🚨 ACCIDENTE CONFIRMADO - Cámara {self.camera_id}")

    def _start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.frames_to_record_after = 375
            self.recording_frames = []
            self.recording_start_time = datetime.now()
            logger.info(f"🎬 Iniciando grabación CON ANOTACIONES - Cámara {self.camera_id}")

    def _save_recording(self):
        try:
            self.is_recording = False
            videos_dir = db.get_config('ruta_videos') or r"C:\Users\Ramirez\Desktop\ACCIDENT\BACKEND\clips"
            os.makedirs(videos_dir, exist_ok=True)
            timestamp = self.recording_start_time.strftime("%Y%m%d_%H%M%S")
            filename = f"cam{self.camera_id}_accident_{timestamp}_ANNOTATED.mp4"
            filepath = os.path.join(videos_dir, filename)
            logger.info(f"💾 Guardando video en: {filepath}")

            all_frames = list(self.frame_buffer) + self.recording_frames
            if len(all_frames) > 0:
                height, width = all_frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(filepath, fourcc, 25.0, (width, height))
                for frame in all_frames:
                    out.write(frame)
                out.release()
                if os.path.exists(filepath):
                    self._save_to_database(filepath)
            self.recording_frames = []
        except Exception as e:
            logger.error(f"❌ Error guardando video: {e}")

    def _save_to_database(self, video_path):
        try:
            accident_id = db.save_accident(
                id_camara=self.camera_id,
                ruta_archivo=video_path,
                latitud=self.camera_data.get("latitud"),
                longitud=self.camera_data.get("longitud"),
                descripcion="Accidente detectado y grabado con IA (anotado)"
            )
            logger.info(f"✅ Accidente guardado en BD - Cámara {self.camera_id}, ID: {accident_id}")
            
            # 🚨 NOTIFICAR A APP MÓVIL EN TIEMPO REAL
            try:
                self.socketio.emit('mobile_emergency_alert', {
                    'accident_id': accident_id,
                    'camera_id': self.camera_id,
                    'camera_ip': self.camera_data.get('ip', 'N/A'),
                    'latitude': self.camera_data.get('latitud', 0),
                    'longitude': self.camera_data.get('longitud', 0),
                    'timestamp': datetime.now().isoformat(),
                    'image_url': f'https://accident-detector.site/api/mobile/image/{accident_id}',
                    'message': f'🚨 Accidente confirmado en cámara {self.camera_data.get("ip", self.camera_id)}',
                    'severity': 'high',
                    'confidence': int(self.last_detection_confidence * 100) if self.last_detection_confidence else 0
                }, room='mobile_emergency')
                
                logger.info(f"📱 Alerta móvil enviada para accidente #{accident_id}")
            except Exception as e:
                logger.error(f"❌ Error enviando alerta móvil: {e}")
                
        except Exception as e:
            logger.error(f"❌ Error guardando en BD: {e}")

    # ==================== EMISIÓN DE FRAMES ====================

    def _emit_frame(self):
        if self.current_frame is None:
            return
        try:
            h, w = self.current_frame.shape[:2]
            if w > 1280:
                scale = 1280 / w
                frame_resized = cv2.resize(self.current_frame, (1280, int(h * scale)))
            else:
                frame_resized = self.current_frame
            _, buffer = cv2.imencode(".jpg", frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            frame_b64 = base64.b64encode(buffer).decode("utf-8")
            self.socketio.emit("camera_frame", {
                "camera_id": str(self.camera_id),
                "frame": frame_b64,
                "is_recording": self.is_recording,
                "timestamp": time.time(),
                "frame_count": self.frame_count
            })
        except Exception as e:
            logger.error(f"❌ Error emitiendo frame cámara {self.camera_id}: {e}")

    def _emit_confirmed(self, confidence, bbox):
        try:
            self.socketio.emit("severe_detected", {
                "camera_id": self.camera_id,
                "camera_ip": self.camera_data.get("ip", "Unknown"),
                "confidence": round(confidence * 100, 2),
                "timestamp": datetime.now().isoformat(),
                "latitud": self.camera_data.get("latitud"),
                "longitud": self.camera_data.get("longitud"),
                "bbox": bbox,
                "status": "CONFIRMADO"
            })
        except Exception as e:
            logger.error(f"❌ Error emitiendo confirmación: {e}")


class CameraManager:
    """Gestor de streams"""
    def __init__(self, socketio, use_yolo=False):
        self.socketio = socketio
        self.use_yolo = use_yolo
        self.active_streams = {}
        logger.info("🎬 CameraManager inicializado")

    def start_camera(self, camera_id):
        if camera_id in self.active_streams:
            return False
        camera_data = db.get_camera_by_id(camera_id)
        if not camera_data:
            return False
        stream = CameraStream(camera_data, self.socketio, use_yolo=self.use_yolo)
        if stream.start():
            self.active_streams[camera_id] = stream
            return True
        return False

    def stop_camera(self, camera_id):
        if camera_id not in self.active_streams:
            return False
        stream = self.active_streams[camera_id]
        stream.stop()
        del self.active_streams[camera_id]
        return True

    def stop_all(self):
        for camera_id in list(self.active_streams.keys()):
            self.stop_camera(camera_id)

    def get_active_cameras(self):
        return list(self.active_streams.keys())

    def is_camera_active(self, camera_id):
        return camera_id in self.active_streams

    def get_camera_stats(self, camera_id):
        if camera_id in self.active_streams:
            s = self.active_streams[camera_id]
            return {
                "total_detections": s.total_detections,
                "confirmed_accidents": s.confirmed_accidents,
                "consecutive_detections": s.consecutive_detections,
                "frame_count": s.frame_count,
                "required_frames": s.required_consecutive,
            }
        return None
