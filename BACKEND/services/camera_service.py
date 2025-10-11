import cv2
import threading
import time
import logging
import base64
from datetime import datetime
from config import Config
from database import db
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Detector lazy-load
_detector_instance = None
def get_detector():
    global _detector_instance
    if _detector_instance is None:
        from models.detector import SevereAccidentDetector
        _detector_instance = SevereAccidentDetector()
    return _detector_instance

class CameraStream:
    """Streaming RTSP con opción de YOLO"""
    def __init__(self, camera_data, socketio, use_yolo=False):
        self.camera_id = camera_data['id']
        self.camera_data = camera_data
        self.socketio = socketio
        self.use_yolo = use_yolo

        self.rtsp_url = self._build_rtsp_url()
        self.is_running = False
        self.thread = None
        self.cap = None
        self.current_frame = None
        self.frame_count = 0
        
        # Control de FPS para emisión
        self.last_emit_time = 0
        self.emit_interval = 0.033  # ~30 FPS

        # Detector
        self.detector = None
        
        # 🎯 SISTEMA DE VERIFICACIÓN DE ACCIDENTES
        self.consecutive_detections = 0
        self.required_consecutive = 5  # Frames consecutivos necesarios
        self.last_detection_time = 0
        self.cooldown_seconds = 3  # Tiempo entre confirmaciones

        logger.info(f"🎥 CameraStream inicializado - ID: {self.camera_id}, URL: {self.rtsp_url}")

    def _build_rtsp_url(self):
        if self.camera_data.get('url_rtsp'):
            return self.camera_data['url_rtsp']
        ip = self.camera_data['ip']
        puerto = self.camera_data.get('puerto', '554')
        usuario = self.camera_data.get('usuario', '')
        password = self.camera_data.get('password', '')
        if usuario and password:
            return f"rtsp://{usuario}:{password}@{ip}:{puerto}/Streaming/Channels/101"
        else:
            return f"rtsp://{ip}:{puerto}/stream"

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
        max_reconnect_attempts = 5
        
        while self.is_running:
            try:
                # Intentar conectar con la cámara
                logger.info(f"🔌 Conectando a cámara {self.camera_id}...")
                self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                
                # Configuraciones para mejor rendimiento
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FPS, 25)
                
                if not self.cap.isOpened():
                    reconnect_attempts += 1
                    logger.error(f"❌ No se pudo abrir cámara {self.camera_id} (intento {reconnect_attempts}/{max_reconnect_attempts})")
                    
                    if reconnect_attempts >= max_reconnect_attempts:
                        logger.error(f"🚫 Máximo de intentos alcanzado para cámara {self.camera_id}")
                        break
                    
                    time.sleep(5)
                    continue
                
                reconnect_attempts = 0
                logger.info(f"✅ Cámara {self.camera_id} conectada exitosamente")

                # Loop principal de captura
                while self.is_running:
                    ret, frame = self.cap.read()
                    
                    if not ret or frame is None:
                        logger.warning(f"⚠️ Frame perdido - Cámara {self.camera_id}")
                        time.sleep(0.1)
                        continue

                    self.frame_count += 1

                    # Procesar con YOLO si está activado
                    if self.use_yolo:
                        if self.detector is None:
                            self.detector = get_detector()
                        tiene_severe, confidence, annotated, bbox = self.detector.detect_severe(frame)
                        self.current_frame = annotated
                        
                        # 🚨 EMITIR DETECCIÓN SEVERE
                        if tiene_severe:
                            self._emit_detection(confidence, bbox)
                    else:
                        self.current_frame = frame

                    # Emitir frame al frontend (con control de FPS)
                    current_time = time.time()
                    if current_time - self.last_emit_time >= self.emit_interval:
                        self._emit_frame()
                        self.last_emit_time = current_time

                    # Log cada 100 frames
                    if self.frame_count % 100 == 0:
                        logger.debug(f"📊 Cámara {self.camera_id}: {self.frame_count} frames procesados")

            except Exception as e:
                logger.error(f"💥 Error en cámara {self.camera_id}: {e}")
                time.sleep(3)
            
            finally:
                if self.cap:
                    self.cap.release()
                    self.cap = None

    def _emit_frame(self):
        """Emitir frame al frontend via Socket.IO"""
        if self.current_frame is None:
            return
        
        try:
            # Redimensionar si es muy grande (optimización)
            h, w = self.current_frame.shape[:2]
            if w > 1280:
                scale = 1280 / w
                frame_resized = cv2.resize(self.current_frame, (1280, int(h * scale)))
            else:
                frame_resized = self.current_frame

            # Comprimir a JPEG con calidad ajustable
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
            _, buffer = cv2.imencode('.jpg', frame_resized, encode_param)
            
            # Convertir a Base64
            frame_b64 = base64.b64encode(buffer).decode('utf-8')

            # Emitir via Socket.IO
            self.socketio.emit('camera_frame', {
                'camera_id': str(self.camera_id),  # Importante: convertir a string
                'frame': frame_b64,
                'is_recording': False,
                'timestamp': time.time(),
                'frame_count': self.frame_count
            })

        except Exception as e:
            logger.error(f"❌ Error emitiendo frame cámara {self.camera_id}: {e}")

    def _emit_detection(self, confidence, bbox):
        """Emitir detección SEVERE al frontend"""
        try:
            camera_data = self.camera_data
            
            detection_data = {
                'camera_id': self.camera_id,
                'camera_ip': camera_data.get('ip', 'Unknown'),
                'confidence': round(confidence * 100, 2),
                'timestamp': datetime.now().isoformat(),
                'latitud': camera_data.get('latitud'),
                'longitud': camera_data.get('longitud'),
                'bbox': bbox if bbox else None
            }
            
            # Emitir al frontend
            self.socketio.emit('severe_detected', detection_data)
            
            # Guardar en BD
            try:
                db.insert_accident(
                    id_camara=self.camera_id,
                    descripcion=f"Accidente SEVERE detectado con {detection_data['confidence']}% confianza",
                    latitud=detection_data['latitud'],
                    longitud=detection_data['longitud']
                )
                logger.info(f"🚨 SEVERE detectado - Cámara {self.camera_id} - Confianza: {detection_data['confidence']}%")
            except Exception as e:
                logger.error(f"Error guardando accidente en BD: {e}")
                
        except Exception as e:
            logger.error(f"Error emitiendo detección: {e}")


class CameraManager:
    """Gestor de streams"""
    def __init__(self, socketio, use_yolo=False):
        self.socketio = socketio
        self.use_yolo = use_yolo
        self.active_streams = {}
        logger.info("🎬 CameraManager inicializado")

    def start_camera(self, camera_id):
        """Iniciar streaming de una cámara"""
        if camera_id in self.active_streams:
            logger.warning(f"⚠️ Cámara {camera_id} ya está activa")
            return False
        
        # Obtener datos de la cámara desde la BD
        camera_data = db.get_camera_by_id(camera_id)
        if not camera_data:
            logger.error(f"❌ Cámara {camera_id} no encontrada en BD")
            return False
        
        # Crear y iniciar stream
        stream = CameraStream(camera_data, self.socketio, use_yolo=self.use_yolo)
        if stream.start():
            self.active_streams[camera_id] = stream
            logger.info(f"✅ Cámara {camera_id} iniciada correctamente")
            return True
        
        logger.error(f"❌ No se pudo iniciar cámara {camera_id}")
        return False

    def stop_camera(self, camera_id):
        """Detener streaming de una cámara"""
        if camera_id not in self.active_streams:
            logger.warning(f"⚠️ Cámara {camera_id} no está activa")
            return False
        
        stream = self.active_streams[camera_id]
        stream.stop()
        del self.active_streams[camera_id]
        logger.info(f"🛑 Cámara {camera_id} detenida")
        return True

    def stop_all(self):
        """Detener todos los streams"""
        logger.info("🛑 Deteniendo todos los streams...")
        for camera_id in list(self.active_streams.keys()):
            self.stop_camera(camera_id)
        logger.info("✅ Todos los streams detenidos")

    def get_active_cameras(self):
        """Obtener lista de IDs de cámaras activas"""
        return list(self.active_streams.keys())
    
    def is_camera_active(self, camera_id):
        """Verificar si una cámara está activa"""
        return camera_id in self.active_streams