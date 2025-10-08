import cv2
import threading
import time
import logging
from datetime import datetime
from config import Config
from database import db
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy loading del detector
_detector_instance = None

def get_detector():
    """Obtener instancia del detector (lazy loading)"""
    global _detector_instance
    if _detector_instance is None:
        from models.detector import SevereAccidentDetector
        _detector_instance = SevereAccidentDetector()
    return _detector_instance

class CameraStream:
    """Maneja el streaming y detección de una cámara individual"""
    
    def __init__(self, camera_data, socketio):
        self.camera_id = camera_data['id']
        self.camera_data = camera_data
        self.socketio = socketio
        
        # Construir URL RTSP
        self.rtsp_url = self._build_rtsp_url()
        
        # Estado del stream
        self.is_running = False
        self.thread = None
        self.cap = None
        
        # Control de detección y grabación
        self.current_frame = None
        self.frame_count = 0
        self.severe_count = 0
        
        # Grabación de clips
        self.is_recording = False
        self.video_writer = None
        self.cooldown_frames = 0
        self.current_clip_path = None
        self.current_accident_id = None
        
        # Detector (se carga cuando se necesite)
        self.detector = None
        
        logger.info(f"CameraStream inicializado - ID: {self.camera_id}, URL: {self.rtsp_url}")
    
    def _build_rtsp_url(self):
        """Construir URL RTSP desde los datos de la cámara"""
        if self.camera_data.get('url_rtsp'):
            return self.camera_data['url_rtsp']
        
        # Construir manualmente si no existe
        ip = self.camera_data['ip']
        puerto = self.camera_data.get('puerto', '554')
        usuario = self.camera_data.get('usuario', '')
        password = self.camera_data.get('password', '')
        
        if usuario and password:
            return f"rtsp://{usuario}:{password}@{ip}:{puerto}/Streaming/Channels/101"
        else:
            return f"rtsp://{ip}:{puerto}/stream"
    
    def start(self):
        """Iniciar el stream de la cámara"""
        if self.is_running:
            logger.warning(f"Stream {self.camera_id} ya está corriendo")
            return False
        
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info(f"✓ Stream iniciado - Cámara ID: {self.camera_id}")
        return True
    
    def stop(self):
        """Detener el stream"""
        self.is_running = False
        
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        if self.cap:
            self.cap.release()
        
        logger.info(f"✓ Stream detenido - Cámara ID: {self.camera_id}")
    
    def _capture_loop(self):
        """Loop principal de captura y detección"""
        import numpy as np

        reconnect_attempts = 0

        while self.is_running:
            try:
                # 🔧 Forzar transporte TCP y pipeline FFMPEG
                rtsp_url = f"{self.rtsp_url}?rtsp_transport=tcp&stimeout=5000000"
                logger.info(f"🎥 Intentando conexión (TCP forzado) a cámara {self.camera_id}: {rtsp_url}")

                # Configurar VideoCapture
                self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))


                if not self.cap.isOpened():
                    reconnect_attempts += 1
                    logger.error(f"❌ No se pudo abrir cámara {self.camera_id} (intento {reconnect_attempts})")

                    if reconnect_attempts >= Config.RTSP_RECONNECT_ATTEMPTS:
                        logger.error(f"🚫 Máximo de intentos alcanzado para cámara {self.camera_id}")
                        break

                    time.sleep(5)
                    continue

                # Conexión exitosa
                reconnect_attempts = 0
                fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                logger.info(f"✅ Cámara {self.camera_id} conectada - {width}x{height} @ {fps}FPS")

                frame_failures = 0

                # 🔁 Loop de lectura
                while self.is_running:
                    ret, frame = self.cap.read()

                    if not ret or frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                        frame_failures += 1
                        logger.warning(f"⚠️ Frame vacío o inválido (#{frame_failures}) cámara {self.camera_id}")

                        # Si falla más de 20 veces seguidas → reinicia conexión
                        if frame_failures > 20:
                            logger.error(f"🔄 Reiniciando stream cámara {self.camera_id} (demasiados frames vacíos)")
                            break
                        time.sleep(0.1)
                        continue

                    # Resetear contador si vuelve el video
                    frame_failures = 0
                    self.frame_count += 1
                    self.current_frame = frame

                    # Procesar cada N frames
                    if self.frame_count % Config.FRAME_SKIP == 0:
                        self._process_frame(frame, fps)

                    # Emitir frame al frontend cada 5 frames
                    if self.frame_count % 5 == 0:
                        self._emit_frame()

                    # Mantener ~30 FPS
                    time.sleep(0.03)

            except Exception as e:
                logger.error(f"💥 Error en capture_loop cámara {self.camera_id}: {e}")
                time.sleep(3)

            finally:
                if self.cap:
                    self.cap.release()
                    self.cap = None


    def _process_frame(self, frame, fps):
        """Procesar frame con detector YOLO"""
        # Cargar detector si no existe
        if self.detector is None:
            self.detector = get_detector()
        
        tiene_severe, confidence, annotated, bbox = self.detector.detect_severe(frame)
        
        self.current_frame = annotated
        
        # Lógica de detección consecutiva
        if tiene_severe:
            self.severe_count += 1
        else:
            self.severe_count = max(0, self.severe_count - 1)
        
        # Iniciar grabación si se supera el umbral
        if self.severe_count >= Config.CONSECUTIVE_THRESHOLD and not self.is_recording:
            self._start_recording(frame, fps, confidence, bbox)
        
        # Continuar grabación
        if self.is_recording:
            self._continue_recording(annotated, fps)
    
    def _start_recording(self, frame, fps, confidence, bbox):
        """Iniciar grabación de clip de accidente"""
        try:
            self.is_recording = True
            self.cooldown_frames = 0
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"severe_cam{self.camera_id}_{timestamp}.{Config.VIDEO_EXTENSION}"
            self.current_clip_path = os.path.join(Config.CLIPS_FOLDER, filename)
            
            # Configurar VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*Config.VIDEO_CODEC)
            height, width = frame.shape[:2]
            self.video_writer = cv2.VideoWriter(self.current_clip_path, fourcc, fps, (width, height))
            
            # Guardar snapshot
            snapshot_path = self.detector.save_snapshot(frame, bbox, confidence, self.camera_id)
            
            # Guardar en BD - Tabla accidentes
            descripcion = f"Accidente SEVERE detectado - Confianza: {confidence:.2f}"
            self.current_accident_id = db.save_accident(
                id_camara=self.camera_id,
                ruta_archivo=self.current_clip_path,
                latitud=self.camera_data.get('latitud'),
                longitud=self.camera_data.get('longitud'),
                descripcion=descripcion
            )
            
            # Guardar evidencia - snapshot
            if snapshot_path:
                db.save_evidence(
                    id_accidente=self.current_accident_id,
                    tipo='imagen',
                    ruta_archivo=snapshot_path,
                    anonimizado=0
                )
            
            logger.info(f"🚨 GRABANDO CLIP - Cámara {self.camera_id} - Archivo: {filename}")
            
            # Emitir alerta al frontend
            self.socketio.emit('severe_detected', {
                'camera_id': self.camera_id,
                'camera_ip': self.camera_data.get('ip'),
                'timestamp': datetime.now().isoformat(),
                'confidence': round(confidence, 2),
                'accident_id': self.current_accident_id,
                'clip_path': self.current_clip_path
            })
            
        except Exception as e:
            logger.error(f"Error iniciando grabación: {e}")
            self.is_recording = False
    
    def _continue_recording(self, annotated_frame, fps):
        """Continuar grabación del clip"""
        if not self.video_writer:
            return
        
        # Escribir frame
        self.video_writer.write(annotated_frame)
        self.cooldown_frames += 1
        
        # Detener si no hay más "severe" y pasó el cooldown
        if self.severe_count == 0 and self.cooldown_frames > (fps * Config.COOLDOWN_SECONDS):
            self._stop_recording()
    
    def _stop_recording(self):
        """Detener grabación del clip"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        self.cooldown_frames = 0
        
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        # Guardar evidencia - video
        if self.current_accident_id and self.current_clip_path:
            db.save_evidence(
                id_accidente=self.current_accident_id,
                tipo='video',
                ruta_archivo=self.current_clip_path,
                anonimizado=0
            )
        
        logger.info(f"✓ Grabación finalizada - Cámara {self.camera_id} - Clip: {self.current_clip_path}")
        
        # Emitir notificación
        self.socketio.emit('recording_finished', {
            'camera_id': self.camera_id,
            'accident_id': self.current_accident_id,
            'clip_path': self.current_clip_path
        })
        
        self.current_accident_id = None
        self.current_clip_path = None
    
    def _emit_frame(self):
        """Enviar frame actual al frontend via WebSocket"""
        if self.current_frame is None:
            return
        
        try:
            import base64
            _, buffer = cv2.imencode('.jpg', self.current_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            self.socketio.emit('camera_frame', {
                'camera_id': self.camera_id,
                'frame': frame_base64,
                'is_recording': self.is_recording
            })
        except Exception as e:
            logger.error(f"Error emitiendo frame: {e}")


class CameraManager:
    """Gestor global de todos los streams de cámaras"""
    
    def __init__(self, socketio):
        self.socketio = socketio
        self.active_streams = {}
        logger.info("CameraManager inicializado")
    
    def start_camera(self, camera_id):
        """Iniciar stream de una cámara"""
        if camera_id in self.active_streams:
            logger.warning(f"Cámara {camera_id} ya está activa")
            return False
        
        # Obtener datos de la cámara desde BD
        camera_data = db.get_camera_by_id(camera_id)
        if not camera_data:
            logger.error(f"Cámara {camera_id} no encontrada en BD")
            return False
        
        # Crear y iniciar stream
        stream = CameraStream(camera_data, self.socketio)
        if stream.start():
            self.active_streams[camera_id] = stream
            return True
        
        return False
    
    def stop_camera(self, camera_id):
        """Detener stream de una cámara"""
        if camera_id not in self.active_streams:
            return False
        
        stream = self.active_streams[camera_id]
        stream.stop()
        del self.active_streams[camera_id]
        return True
    
    def stop_all(self):
        """Detener todos los streams"""
        for camera_id in list(self.active_streams.keys()):
            self.stop_camera(camera_id)
        logger.info("Todos los streams detenidos")
    
    def get_active_cameras(self):
        """Obtener lista de cámaras activas"""
        return list(self.active_streams.keys())