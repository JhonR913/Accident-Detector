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
        # ============================================
        # VERIFICAR Y CONFIGURAR GPU
        # ============================================
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print("\n" + "="*70)
        print("🎮 DETECTOR DE ACCIDENTES SEVERE")
        print("="*70)
        print(f"Dispositivo: {self.device.upper()}")
        
        if self.device == 'cuda':
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✅ VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            print(f"✅ CUDA Version: {torch.version.cuda}")
        else:
            print("⚠️ GPU NO DISPONIBLE - Usando CPU (será MUY lento)")
            print("⚠️ Instala PyTorch con CUDA:")
            print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        
        print("="*70 + "\n")
        
        # ============================================
        # CONFIGURAR SAFE GLOBALS (PyTorch 2.6+)
        # ============================================
        logger.info(f"Cargando modelo YOLO desde {Config.YOLO_MODEL_PATH}...")
        
        try:
            import torch.nn as nn
            from collections import OrderedDict
            
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
            
            try:
                from ultralytics.nn.tasks import DetectionModel
                safe_classes.append(DetectionModel)
            except:
                pass
            
            torch.serialization.add_safe_globals(safe_classes)
            logger.info("✓ Safe globals configurados para PyTorch 2.6+")
        except Exception as e:
            logger.warning(f"Advertencia configurando safe globals: {e}")
        
        # ============================================
        # CARGAR MODELO Y MOVERLO A GPU
        # ============================================
        self.model = YOLO(Config.YOLO_MODEL_PATH)
        self.model.to(self.device)  # ✅ FORZAR GPU
        
        self.confidence = Config.CONFIDENCE_THRESHOLD
        self.target_class = Config.TARGET_CLASS
        
        # Verificar clases disponibles
        available_classes = list(self.model.names.values())
        logger.info(f"Clases del modelo: {available_classes}")
        
        if self.target_class not in available_classes:
            logger.warning(f"⚠️ Clase '{self.target_class}' NO encontrada")
            logger.warning(f"Usando primera clase: {available_classes[0] if available_classes else 'ninguna'}")
        
        logger.info(f"✅ Modelo cargado en {self.device.upper()} - Detectando: '{self.target_class}' (conf >= {self.confidence})")
    
    def detect_severe(self, frame):
        """Detectar accidentes SEVERE en un frame"""
        try:
            # ✅ INFERENCIA CON GPU
            results = self.model.predict(
                frame,
                conf=self.confidence,
                device=self.device,  # ✅ Usar GPU
                half=True,  # ✅ FP16 (2x más rápido en GPU)
                verbose=False
            )
            
            tiene_severe = False
            max_confidence = 0.0
            bbox = None
            
            # Procesar resultados
            for box, conf, cls in zip(
                results[0].boxes.xyxy, 
                results[0].boxes.conf, 
                results[0].boxes.cls
            ):
                class_name = self.model.names[int(cls)]
                
                if class_name == self.target_class:
                    tiene_severe = True
                    confidence_val = float(conf)
                    
                    if confidence_val > max_confidence:
                        max_confidence = confidence_val
                        bbox = box.cpu().numpy().tolist()  # Mover a CPU para guardar
            
            # Anotar frame
            if tiene_severe:
                annotated_frame = results[0].plot()
                cv2.putText(
                    annotated_frame, 
                    "ACCIDENTE SEVERE", 
                    (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1.5, 
                    (0, 0, 255), 
                    3, 
                    cv2.LINE_AA
                )
            else:
                annotated_frame = frame.copy()
            
            return tiene_severe, max_confidence, annotated_frame, bbox
            
        except Exception as e:
            logger.error(f"❌ Error en detect_severe: {e}")
            import traceback
            traceback.print_exc()
            return False, 0.0, frame, None
    
    def save_snapshot(self, frame, bbox, confidence, camera_id):
        """Guardar snapshot de detección"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"severe_cam{camera_id}_{timestamp}.jpg"
            filepath = os.path.join(Config.SNAPSHOTS_FOLDER, filename)
            
            # Crear directorio si no existe
            os.makedirs(Config.SNAPSHOTS_FOLDER, exist_ok=True)
            
            # Dibujar bounding box
            if bbox:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                
                # Agregar texto con confianza
                text = f"SEVERE: {confidence:.2%}"
                cv2.putText(
                    frame, 
                    text, 
                    (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (0, 0, 255), 
                    2
                )
            
            cv2.imwrite(filepath, frame)
            logger.info(f"💾 Snapshot guardado: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Error guardando snapshot: {e}")
            return None
    
    def process_video_file(self, video_path, output_dir=None):
        """Procesar video completo y detectar accidentes"""
        detections = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"No se pudo abrir: {video_path}")
            
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_number = 0
            
            logger.info(f"📹 Procesando video: {video_path}")
            logger.info(f"   FPS: {fps} | Frames totales: {total_frames}")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Procesar cada N frames (según Config.FRAME_SKIP)
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
                        logger.info(f"🚨 SEVERE en frame {frame_number} ({int(ts//60):02d}:{int(ts%60):02d}) - Conf: {confidence:.2%}")
                
                frame_number += 1
                
                # Log de progreso cada 100 frames
                if frame_number % 100 == 0:
                    progress = (frame_number / total_frames) * 100
                    logger.info(f"⏳ Progreso: {progress:.1f}% ({frame_number}/{total_frames})")
            
            cap.release()
            logger.info(f"✅ Procesamiento completo: {len(detections)} detecciones SEVERE")
            
        except Exception as e:
            logger.error(f"❌ Error procesando video: {e}")
            import traceback
            traceback.print_exc()
        
        return detections