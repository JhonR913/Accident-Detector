import os
from dotenv import load_dotenv

load_dotenv()

# Obtener directorio base del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    # Base de datos
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_USER = os.getenv('DB_USER', 'root')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')
    DB_NAME = os.getenv('DB_NAME', 'accident_detection')
    
    # YOLO - RUTA ABSOLUTA
    YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best.pt')  # ← CAMBIO AQUÍ
    TARGET_CLASS = 'severe'
    CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))
    CONSECUTIVE_THRESHOLD = 3
    COOLDOWN_SECONDS = 2
    
    # Rutas de almacenamiento - RUTAS ABSOLUTAS
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    CLIPS_FOLDER = os.path.join(BASE_DIR, 'clips')
    SNAPSHOTS_FOLDER = os.path.join(BASE_DIR, 'snapshots')
    
    # Servidor
    HOST = '0.0.0.0'
    PORT = 5000
    DEBUG = True
    
    # RTSP
    RTSP_RECONNECT_ATTEMPTS = 3
    RTSP_TIMEOUT = 10
    FRAME_SKIP = 2
    
    # Video output
    VIDEO_CODEC = 'mp4v'
    VIDEO_EXTENSION = 'mp4'
    
    @staticmethod
    def init_folders():
        """Crear carpetas necesarias"""
        for folder in [Config.UPLOAD_FOLDER, Config.CLIPS_FOLDER, Config.SNAPSHOTS_FOLDER]:
            os.makedirs(folder, exist_ok=True)