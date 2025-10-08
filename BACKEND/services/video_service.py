import os
import logging
from datetime import datetime
from config import Config
from database import db
import shutil

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

class VideoService:
    """Servicio para análisis de videos subidos"""
    
    @staticmethod
    def save_uploaded_video(file):
        """Guardar video subido"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"upload_{timestamp}_{file.filename}"
            filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
            
            file.save(filepath)
            logger.info(f"✓ Video guardado: {filepath}")
            return filepath, filename
        except Exception as e:
            logger.error(f"Error guardando video: {e}")
            return None, None
    
    @staticmethod
    def analyze_video(video_path, filename):
        """
        Analizar video completo buscando accidentes SEVERE y guardar frames anotados
        """
        try:
            import cv2
            logger.info(f"Iniciando análisis de: {video_path}")
        
        # Obtener detector
            detector = get_detector()
        
        # Crear carpeta para frames
            frames_dir = os.path.join(Config.UPLOAD_FOLDER, f"frames_{int(datetime.now().timestamp())}")
            os.makedirs(frames_dir, exist_ok=True)
        
        # Procesar video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"No se pudo abrir: {video_path}")
        
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Crear video anotado
            output_video_path = os.path.join(Config.UPLOAD_FOLDER, f"anotado_{filename}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
            detections = []
            frame_number = 0
            detection_frames = []
        
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
            
                if frame_number % Config.FRAME_SKIP == 0:
                    tiene_severe, confidence, annotated, bbox = detector.detect_severe(frame)
                
                    if tiene_severe:
                        ts = frame_number / fps
                    
                    # Guardar frame anotado como imagen
                        frame_filename = f"detection_{frame_number}.jpg"
                        frame_path = os.path.join(frames_dir, frame_filename)
                        cv2.imwrite(frame_path, annotated)
                    
                        detections.append({
                            'frame': frame_number,
                            'timestamp': f"{int(ts//60):02d}:{int(ts%60):02d}",
                            'confidence': round(confidence, 3),
                            'bbox': bbox,
                            'frame_path': frame_path
                        })
                    
                        detection_frames.append(frame_filename)
                       
                        logger.info(f"SEVERE en frame {frame_number} - Conf: {confidence:.2f}")
                    
                    # Escribir frame anotado 
                        out.write(annotated)
                    else:
                    # Escribir frame original
                        out.write(frame)
                else:
                    out.write(frame)
            
                frame_number += 1
        
            cap.release()
            out.release()
        
            result = {
                'filename': filename,
                'video_path': video_path,
                'annotated_video_path': output_video_path,
                'total_detections': len(detections),
                'detections': detections,
                'detection_frames': detection_frames,
                'frames_dir': frames_dir,
                'analyzed_at': datetime.now().isoformat()
            }
        
            logger.info(f"✓ Análisis completado - {len(detections)} detecciones")
            return result
        
        except Exception as e:
            logger.error(f"Error analizando video: {e}")
            return {
                'error': str(e),
                'filename': filename,
                'total_detections': 0,
                'detections': []
            }
    @staticmethod
    def generate_report(analysis_result):
        """Generar reporte de texto del análisis"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_filename = f"reporte_{timestamp}.txt"
            report_path = os.path.join(Config.UPLOAD_FOLDER, report_filename)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("="*60 + "\n")
                f.write("REPORTE DE ANÁLISIS - DETECCIÓN DE ACCIDENTES SEVERE\n")
                f.write("="*60 + "\n\n")
                f.write(f"Archivo: {analysis_result['filename']}\n")
                f.write(f"Fecha de análisis: {analysis_result['analyzed_at']}\n")
                f.write(f"Total de detecciones SEVERE: {analysis_result['total_detections']}\n\n")
                
                if analysis_result['total_detections'] > 0:
                    f.write("-"*60 + "\n")
                    f.write("DETALLE DE DETECCIONES:\n")
                    f.write("-"*60 + "\n\n")
                    
                    for idx, det in enumerate(analysis_result['detections'], 1):
                        f.write(f"{idx}. ACCIDENTE SEVERE\n")
                        f.write(f"   Timestamp: {det['timestamp']}\n")
                        f.write(f"   Frame: {det['frame']}\n")
                        f.write(f"   Confianza: {det['confidence']:.3f}\n")
                        f.write(f"   BBox: {det['bbox']}\n\n")
                else:
                    f.write("No se detectaron accidentes SEVERE en el video.\n")
            
            logger.info(f"✓ Reporte generado: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Error generando reporte: {e}")
            return None
    
    @staticmethod
    def cleanup_old_files(days=7):
        """Limpiar archivos antiguos de uploads"""
        try:
            import time
            now = time.time()
            cutoff = now - (days * 86400)
            
            removed = 0
            for folder in [Config.UPLOAD_FOLDER, Config.CLIPS_FOLDER]:
                if not os.path.exists(folder):
                    continue
                    
                for filename in os.listdir(folder):
                    filepath = os.path.join(folder, filename)
                    if os.path.isfile(filepath):
                        file_time = os.path.getmtime(filepath)
                        if file_time < cutoff:
                            os.remove(filepath)
                            removed += 1
            
            logger.info(f"✓ Limpieza completada - {removed} archivos eliminados")
            return removed
            
        except Exception as e:
            logger.error(f"Error en limpieza: {e}")
            return 0