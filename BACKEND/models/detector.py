from ultralytics import YOLO
from config import Config
import cv2
import logging
from datetime import datetime
import os
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SevereAccidentDetector:
    def __init__(self):
        logger.info(f"Cargando modelo YOLO desde {Config.YOLO_MODEL_PATH}...")
        
        # Agregar TODAS las clases necesarias para PyTorch 2.6
        try:
            import torch.nn as nn
            from collections import OrderedDict
            
            # Lista completa de clases seguras necesarias
            safe_classes = [
                torch.nn.modules.container.Sequential,
                torch.nn.modules.conv.Conv2d,
                torch.nn.modules.batchnorm.BatchNorm2d,
                torch.nn.modules.activation.SiLU,
                torch.nn.modules.pooling.MaxPool2d,
                torch.nn.modules.upsampling.Upsample,
                torch.nn.modules.linear.Linear,
                OrderedDict,
            ]
            
            # Intentar agregar clases de Ultralytics
            try:
                from ultralytics.nn.tasks import DetectionModel
                safe_classes.append(DetectionModel)
            except:
                pass
            
            torch.serialization.add_safe_globals(safe_classes)
            logger.info("✓ Safe globals configurados para PyTorch 2.6+")
        except Exception as e:
            logger.warning(f"Advertencia configurando safe globals: {e}")
        
        # Cargar modelo
        self.model = YOLO(Config.YOLO_MODEL_PATH)
        self.confidence = Config.CONFIDENCE_THRESHOLD
        self.target_class = Config.TARGET_CLASS
        
        # Verificar clases disponibles
        available_classes = list(self.model.names.values())
        logger.info(f"Clases del modelo: {available_classes}")
        
        if self.target_class not in available_classes:
            logger.warning(f"⚠️ Clase '{self.target_class}' NO encontrada en el modelo")
            logger.warning(f"Usando primera clase disponible: {available_classes[0] if available_classes else 'ninguna'}")
        
        logger.info(f"✓ Modelo cargado - Detectando: '{self.target_class}' (conf >= {self.confidence})")
    
    def detect_severe(self, frame):
        try:
            results = self.model(frame, conf=self.confidence, verbose=False)
            
            tiene_severe = False
            max_confidence = 0.0
            bbox = None
            
            for box, conf, cls in zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls):
                class_name = self.model.names[int(cls)]
                
                if class_name == self.target_class:
                    tiene_severe = True
                    confidence_val = float(conf)
                    
                    if confidence_val > max_confidence:
                        max_confidence = confidence_val
                        bbox = box.cpu().numpy().tolist()
            
            if tiene_severe:
                annotated_frame = results[0].plot()
                cv2.putText(annotated_frame, "ACCIDENTE SEVERE", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                           1.5, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                annotated_frame = frame.copy()
            
            return tiene_severe, max_confidence, annotated_frame, bbox
        except Exception as e:
            logger.error(f"Error en detect_severe: {e}")
            return False, 0.0, frame, None
    
    def save_snapshot(self, frame, bbox, confidence, camera_id):
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"severe_cam{camera_id}_{timestamp}.jpg"
            filepath = os.path.join(Config.SNAPSHOTS_FOLDER, filename)
            
            if bbox:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            cv2.imwrite(filepath, frame)
            logger.info(f"Snapshot: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error snapshot: {e}")
            return None
    
    def process_video_file(self, video_path, output_dir=None):
        detections = []
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"No se pudo abrir: {video_path}")
            
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_number = 0
            
            logger.info(f"Procesando video: {video_path}")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_number % Config.FRAME_SKIP == 0:
                    tiene_severe, confidence, _, bbox = self.detect_severe(frame)
                    
                    if tiene_severe:
                        ts = frame_number / fps
                        detections.append({
                            'frame': frame_number,
                            'timestamp': f"{int(ts//60):02d}:{int(ts%60):02d}",
                            'confidence': round(confidence, 3),
                            'bbox': bbox
                        })
                        logger.info(f"SEVERE en frame {frame_number} - Conf: {confidence:.2f}")
                
                frame_number += 1
            
            cap.release()
            logger.info(f"✓ Procesado: {len(detections)} detecciones")
        except Exception as e:
            logger.error(f"Error procesando: {e}")
        
        return detections