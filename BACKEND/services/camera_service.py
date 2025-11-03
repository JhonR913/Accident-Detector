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
    """Streaming RTSP con opción de YOLO y grabación de evidencia"""
    def __init__(self, camera_data, socketio, use_yolo=True):
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
        
        # 🎯 SISTEMA DE VERIFICACIÓN ROBUSTO
        self.consecutive_detections = 0
        self.required_consecutive = 120  # ✅ 25 frames = ~1 segundo (confirmación sólida)
        self.last_confirmed_time = 0
        self.cooldown_seconds = 60  # ✅ 30 segundos entre confirmaciones
        
        # 📊 Estadísticas
        self.total_detections = 0
        self.confirmed_accidents = 0
        
        # 🎬 BUFFER DE FRAMES ANOTADOS PARA GRABACIÓN (15 segundos antes)
        # A 25 FPS, 15 segundos = 375 frames
        self.frame_buffer = deque(maxlen=375)  # ✅ Buffer con frames ANOTADOS
        self.is_recording = False
        self.frames_to_record_after = 0
        self.recording_frames = []
        self.recording_start_time = None
        
        # 🎨 ALMACENAR ÚLTIMO BBOX PARA GRABACIÓN
        self.last_detection_bbox = None
        self.last_detection_confidence = 0

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
        max_reconnect_attempts = 10

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
                    logger.warning(f"❌ No se pudo abrir cámara {self.camera_id}. Reintentando ({reconnect_attempts}/{max_reconnect_attempts})...")
                    time.sleep(3)
                    continue

                logger.info(f"✅ Cámara {self.camera_id} conectada con FFmpeg optimizado")
                reconnect_attempts = 0

                while self.is_running and self.cap.isOpened():
                    ret, frame = self.cap.read()

                    if not ret or frame is None:
                        logger.warning(f"⚠️ Frame perdido - Cámara {self.camera_id}")
                        time.sleep(0.05)
                        continue

                    self.frame_count += 1
                    
                    # Procesar con YOLO
                    if self.use_yolo:
                        if self.detector is None:
                            self.detector = get_detector()
                        tiene_severe, confidence, annotated, bbox = self.detector.detect_severe(frame)
                        
                        # ✅ USAR FRAME ANOTADO (con bounding box)
                        self.current_frame = annotated
                        
                        # 🎬 GUARDAR FRAME ANOTADO EN BUFFER (siempre)
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
                    
                    # 🎬 Si está grabando, agregar frame ANOTADO
                    if self.is_recording:
                        self.recording_frames.append(self.current_frame.copy())
                        self.frames_to_record_after -= 1
                        
                        # Finalizar grabación
                        if self.frames_to_record_after <= 0:
                            self._save_recording()

                    # Emitir frame
                    now = time.time()
                    if now - self.last_emit_time >= self.emit_interval:
                        self._emit_frame()
                        self.last_emit_time = now

                    if self.frame_count % 200 == 0:
                        logger.info(f"📊 Cámara {self.camera_id}: {self.frame_count} frames procesados")

            except Exception as e:
                logger.error(f"💥 Error en cámara {self.camera_id}: {e}")
                import traceback
                logger.error(traceback.format_exc())

            finally:
                if self.cap:
                    self.cap.release()
                    self.cap = None

                if self.is_running:
                    logger.warning(f"♻️ Intentando reconectar cámara {self.camera_id} en 2s...")
                    time.sleep(2)

    def _verify_detection(self, confidence, bbox):
        """🎯 Sistema de verificación optimizado para tiempo real"""
        current_time = time.time()
        
        self.consecutive_detections += 1
        self.total_detections += 1
        
        # Emitir tentativa con progreso visual
        try:
            progress_percentage = (self.consecutive_detections / self.required_consecutive) * 100
            
            self.socketio.emit('tentative_detection', {
                'camera_id': self.camera_id,
                'camera_ip': self.camera_data.get('ip', 'Unknown'),
                'confidence': round(confidence * 100, 2),
                'consecutive_frames': self.consecutive_detections,
                'required_frames': self.required_consecutive,
                'progress': min(progress_percentage, 100),
                'timestamp': datetime.now().isoformat(),
                'status': 'VERIFICANDO',
                'bbox': bbox
            })
        except Exception as e:
            logger.error(f"Error emitiendo tentativa: {e}")
        
        # ✅ CONFIRMAR: Solo necesita 3 frames consecutivos + cooldown
        if (self.consecutive_detections >= self.required_consecutive and 
            current_time - self.last_confirmed_time >= self.cooldown_seconds):
            
            self.confirmed_accidents += 1
            self.last_confirmed_time = current_time
            self.consecutive_detections = 0
            
            # 🎬 Iniciar grabación de 15 segundos después
            self._start_recording()
            
            # Emitir confirmación
            self._emit_confirmed(confidence, bbox)
            
            logger.warning(f"🚨 ACCIDENTE CONFIRMADO en {self.required_consecutive} frames (~{self.required_consecutive * 0.04:.2f}s) - Cámara {self.camera_id}")

    def _start_recording(self):
        """🎬 Iniciar grabación de 15 segundos después del accidente"""
        if not self.is_recording:
            self.is_recording = True
            self.frames_to_record_after = 375  # 15 segundos a 25 FPS
            self.recording_frames = []
            self.recording_start_time = datetime.now()
            logger.info(f"🎬 Iniciando grabación CON ANOTACIONES - Cámara {self.camera_id}")

    def _save_recording(self):
        """💾 Guardar video completo CON ANOTACIONES (15 seg antes + 15 seg después)"""
        try:
            self.is_recording = False
            
            videos_dir = r"C:\Users\Ramirez\Desktop\ACCIDENT\BACKEND\clips"
            os.makedirs(videos_dir, exist_ok=True)
            
            timestamp = self.recording_start_time.strftime("%Y%m%d_%H%M%S")
            filename = f"cam{self.camera_id}_accident_{timestamp}_ANNOTATED.mp4"
            filepath = os.path.join(videos_dir, filename)
            
            logger.info(f"💾 Guardando video ANOTADO en: {filepath}")
            
            # ✅ COMBINAR FRAMES ANOTADOS: buffer + recording_frames
            all_frames = list(self.frame_buffer) + self.recording_frames
            
            if len(all_frames) > 0:
                height, width = all_frames[0].shape[:2]
                
                # Crear VideoWriter con mejor codec
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(filepath, fourcc, 25.0, (width, height))
                
                if not out.isOpened():
                    logger.error(f"❌ No se pudo crear VideoWriter")
                    return
                
                # 🎨 Escribir frames con anotaciones
                frames_written = 0
                for frame in all_frames:
                    out.write(frame)
                    frames_written += 1
                
                out.release()
                
                logger.info(f"✅ Video ANOTADO guardado: {filepath}")
                logger.info(f"   📊 Frames: {frames_written}, Duración: ~{frames_written/25:.1f}s")
                
                # Verificar archivo
                if os.path.exists(filepath):
                    file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
                    logger.info(f"   💾 Tamaño: {file_size:.2f} MB")
                    
                    # Guardar en BD
                    self._save_to_database(filepath)
                else:
                    logger.error(f"❌ El archivo no existe después de guardarlo")
                
            else:
                logger.warning(f"⚠️ No hay frames para guardar - Cámara {self.camera_id}")
            
            self.recording_frames = []
            
        except Exception as e:
            logger.error(f"❌ Error guardando video: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _save_to_database(self, video_path):
        """💾 Guardar accidente en BD con ruta del video"""
        try:
            accident_id = db.save_accident(
                id_camara=self.camera_id,
                ruta_archivo=video_path,
                latitud=self.camera_data.get('latitud'),
                longitud=self.camera_data.get('longitud'),
                descripcion=f"Accidente SEVERE confirmado en {self.required_consecutive} frames - Video ANOTADO de 30s con bounding boxes"
            )
            
            logger.info(f"✅ Accidente guardado en BD - ID: {accident_id}")
            
        except Exception as e:
            logger.error(f"❌ Error guardando en BD: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _emit_frame(self):
        """Emitir frame al frontend via Socket.IO"""
        if self.current_frame is None:
            return
        
        try:
            h, w = self.current_frame.shape[:2]
            if w > 1280:
                scale = 1280 / w
                frame_resized = cv2.resize(self.current_frame, (1280, int(h * scale)))
            else:
                frame_resized = self.current_frame

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
            _, buffer = cv2.imencode('.jpg', frame_resized, encode_param)
            
            frame_b64 = base64.b64encode(buffer).decode('utf-8')

            self.socketio.emit('camera_frame', {
                'camera_id': str(self.camera_id),
                'frame': frame_b64,
                'is_recording': self.consecutive_detections > 0 or self.is_recording,
                'timestamp': time.time(),
                'frame_count': self.frame_count,
                'yolo_active': self.use_yolo,
                'consecutive_detections': self.consecutive_detections,
                'detection_progress': (self.consecutive_detections / self.required_consecutive) * 100
            })

        except Exception as e:
            logger.error(f"❌ Error emitiendo frame cámara {self.camera_id}: {e}")

    def _emit_confirmed(self, confidence, bbox):
        """🚨 Emitir accidente CONFIRMADO al frontend"""
        try:
            detection_data = {
                'camera_id': self.camera_id,
                'camera_ip': self.camera_data.get('ip', 'Unknown'),
                'confidence': round(confidence * 100, 2),
                'timestamp': datetime.now().isoformat(),
                'latitud': self.camera_data.get('latitud'),
                'longitud': self.camera_data.get('longitud'),
                'bbox': bbox if bbox else None,
                'status': 'CONFIRMADO',
                'total_detections': self.total_detections,
                'confirmed_count': self.confirmed_accidents,
                'verification_time': f"{self.required_consecutive * 0.04:.2f}s"
            }
            
            self.socketio.emit('severe_detected', detection_data)
            logger.info(f"✅ Confirmación emitida - Cámara {self.camera_id}")
                
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
        """Iniciar streaming de una cámara"""
        if camera_id in self.active_streams:
            logger.warning(f"⚠️ Cámara {camera_id} ya está activa")
            return False
        
        camera_data = db.get_camera_by_id(camera_id)
        if not camera_data:
            logger.error(f"❌ Cámara {camera_id} no encontrada en BD")
            return False
        
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
    
    def get_camera_stats(self, camera_id):
        """Obtener estadísticas de una cámara"""
        if camera_id in self.active_streams:
            stream = self.active_streams[camera_id]
            return {
                'total_detections': stream.total_detections,
                'confirmed_accidents': stream.confirmed_accidents,
                'consecutive_detections': stream.consecutive_detections,
                'frame_count': stream.frame_count,
                'required_frames': stream.required_consecutive,
                'cooldown_seconds': stream.cooldown_seconds
            }
        return None