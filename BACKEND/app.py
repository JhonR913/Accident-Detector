from flask import Flask, request, jsonify, send_file, send_from_directory, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import logging
import os
import cv2
from datetime import datetime

from config import Config
from database import db
from services.camera_service import CameraManager
from services.video_service import VideoService

FRONTEND_DIR = r"C:\Users\Ramirez\Desktop\ACCIDENT\FRONTED"

# Inicializar Flask
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')

# Configuración
app.config['SECRET_KEY'] = 'accident-detection-secret-key'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# CORS
CORS(app, resources={r"/*": {"origins": "*"}})

# Socket.IO con configuración optimizada
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',  # Cambiar de 'eventlet' a 'threading'
    logger=True,
    engineio_logger=False,  # Reducir ruido en logs
    ping_timeout=60,
    ping_interval=25,
    max_http_buffer_size=10000000  # 10MB para frames grandes
)

# Inicializar servicios
Config.init_folders()
camera_manager = CameraManager(socketio, use_yolo=  True)

logger.info("=" * 60)
logger.info("🚨 SISTEMA DE DETECCIÓN DE ACCIDENTES INICIADO")
logger.info("=" * 60)

# ============================================
# WEBSOCKET EVENTS
# ============================================

@socketio.on('connect')
def handle_connect():
    logger.info(f"✅ Cliente conectado: {request.sid}")
    emit('connection_response', {
        'status': 'connected',
        'message': 'Conexión establecida con el servidor',
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"❌ Cliente desconectado: {request.sid}")

@socketio.on('ping')
def handle_ping():
    """Responder a ping del cliente"""
    emit('pong', {'timestamp': datetime.now().isoformat()})

@socketio.on('request_camera_list')
def handle_camera_list_request():
    """Cliente solicita lista de cámaras"""
    try:
        cameras = db.get_all_cameras()
        active = camera_manager.get_active_cameras()
        
        for cam in cameras:
            cam['is_streaming'] = cam['id'] in active
        
        emit('camera_list', {'cameras': cameras})
    except Exception as e:
        logger.error(f"Error enviando lista de cámaras: {e}")
        emit('error', {'message': str(e)})

# ============================================
# ENDPOINTS - CÁMARAS
# ============================================

@app.route('/api/cameras', methods=['GET'])
def get_cameras():
    """Obtener todas las cámaras"""
    try:
        cameras = db.get_all_cameras()
        active = camera_manager.get_active_cameras()
        
        # Agregar estado activo
        for cam in cameras:
            cam['is_streaming'] = cam['id'] in active
        
        logger.info(f"📹 Enviando {len(cameras)} cámaras ({len(active)} activas)")
        return jsonify({'success': True, 'cameras': cameras})
    except Exception as e:
        logger.error(f"Error obteniendo cámaras: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/cameras/<int:camera_id>', methods=['GET'])
def get_camera(camera_id):
    """Obtener una cámara específica"""
    try:
        camera = db.get_camera_by_id(camera_id)
        if not camera:
            return jsonify({'success': False, 'error': 'Cámara no encontrada'}), 404
        
        return jsonify({'success': True, 'camera': camera})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/cameras', methods=['POST'])
def add_camera():
    """Agregar nueva cámara"""
    try:
        data = request.json
        
        required = ['ip', 'puerto', 'usuario', 'password', 'latitud', 'longitud', 'url_rtsp']
        if not all(k in data for k in required):
            return jsonify({'success': False, 'error': 'Faltan campos requeridos'}), 400
        
        camera_id = db.add_camera(
            ip=data['ip'],
            puerto=data['puerto'],
            usuario=data['usuario'],
            password=data['password'],
            latitud=data['latitud'],
            longitud=data['longitud'],
            url_rtsp=data['url_rtsp']
        )
        
        logger.info(f"✅ Cámara agregada - ID: {camera_id}, IP: {data['ip']}")
        return jsonify({'success': True, 'camera_id': camera_id, 'message': 'Cámara agregada correctamente'})
    
    except Exception as e:
        logger.error(f"Error agregando cámara: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/cameras/<int:camera_id>', methods=['DELETE'])
def delete_camera(camera_id):
    """Eliminar cámara"""
    try:
        # Detener stream si está activo
        if camera_manager.is_camera_active(camera_id):
            camera_manager.stop_camera(camera_id)
        
        db.delete_camera(camera_id)
        logger.info(f"🗑️ Cámara {camera_id} eliminada")
        return jsonify({'success': True, 'message': 'Cámara eliminada'})
    except Exception as e:
        logger.error(f"Error eliminando cámara: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================
# ENDPOINTS - STREAMING
# ============================================

@app.route('/api/cameras/<int:camera_id>/start', methods=['POST'])
def start_camera_stream(camera_id):
    """Iniciar stream de cámara"""
    try:
        logger.info(f"▶️ Iniciando stream - Cámara {camera_id}")
        success = camera_manager.start_camera(camera_id)
        
        if success:
            return jsonify({'success': True, 'message': f'Stream iniciado - Cámara {camera_id}'})
        else:
            return jsonify({'success': False, 'error': 'No se pudo iniciar el stream'}), 500
    except Exception as e:
        logger.error(f"Error iniciando stream: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/cameras/<int:camera_id>/stop', methods=['POST'])
def stop_camera_stream(camera_id):
    """Detener stream de cámara"""
    try:
        logger.info(f"⏹️ Deteniendo stream - Cámara {camera_id}")
        success = camera_manager.stop_camera(camera_id)
        
        if success:
            return jsonify({'success': True, 'message': f'Stream detenido - Cámara {camera_id}'})
        else:
            return jsonify({'success': False, 'error': 'Stream no estaba activo'}), 400
    except Exception as e:
        logger.error(f"Error deteniendo stream: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================
# ENDPOINTS - ACCIDENTES
# ============================================

@app.route('/api/accidents', methods=['GET'])
def get_accidents():
    """Obtener todos los accidentes"""
    try:
        limit = request.args.get('limit', 100, type=int)
        accidents = db.get_all_accidents(limit)
        return jsonify({'success': True, 'accidents': accidents, 'total': len(accidents)})
    except Exception as e:
        logger.error(f"Error obteniendo accidentes: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/accidents/camera/<int:camera_id>', methods=['GET'])
def get_accidents_by_camera(camera_id):
    """Obtener accidentes de una cámara específica"""
    try:
        limit = request.args.get('limit', 100, type=int)
        accidents = db.get_accidents_by_camera(camera_id, limit)
        return jsonify({'success': True, 'accidents': accidents, 'total': len(accidents)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/accidents/export', methods=['GET'])
def export_accidents_csv():
    """Exportar accidentes a CSV"""
    try:
        import csv
        import io
        
        accidents = db.get_all_accidents(1000)
        
        # Crear CSV en memoria
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Headers
        writer.writerow(['ID', 'Cámara', 'Fecha', 'Latitud', 'Longitud', 'Descripción', 'Archivo'])
        
        # Datos
        for acc in accidents:
            writer.writerow([
                acc['id'],
                acc.get('camera_ip', 'N/A'),
                acc['fecha_accidente'],
                acc.get('latitud', ''),
                acc.get('longitud', ''),
                acc.get('descripcion', ''),
                acc.get('ruta_archivo', '')
            ])
        
        output.seek(0)
        
        return output.getvalue(), 200, {
            'Content-Type': 'text/csv',
            'Content-Disposition': f'attachment; filename=accidentes_{datetime.now().strftime("%Y%m%d")}.csv'
        }
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================
# ENDPOINTS - VIDEOS
# ============================================

@app.route('/api/videos/upload', methods=['POST'])
def upload_video():
    """Subir video para análisis"""
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No se envió ningún video'}), 400
        
        file = request.files['video']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Archivo vacío'}), 400
        
        # Guardar video
        video_path, filename = VideoService.save_uploaded_video(file)
        
        if not video_path:
            return jsonify({'success': False, 'error': 'Error guardando video'}), 500
        
        return jsonify({
            'success': True,
            'message': 'Video subido correctamente',
            'filename': filename,
            'video_path': video_path
        })
    
    except Exception as e:
        logger.error(f"Error subiendo video: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/videos/analyze', methods=['POST'])
def analyze_video():
    """Analizar video subido"""
    try:
        data = request.json
        video_path = data.get('video_path')
        filename = data.get('filename')
        
        if not video_path or not os.path.exists(video_path):
            return jsonify({'success': False, 'error': 'Video no encontrado'}), 404
        
        # Analizar
        result = VideoService.analyze_video(video_path, filename)
        
        # Generar reporte
        report_path = VideoService.generate_report(result)
        
        result['report_path'] = report_path
        
        return jsonify({'success': True, 'analysis': result})
    
    except Exception as e:
        logger.error(f"Error analizando video: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/videos/download-annotated', methods=['POST'])
def download_annotated_video():
    """Descargar video con anotaciones"""
    try:
        data = request.json
        video_path = data.get('video_path')
        
        if not video_path or not os.path.exists(video_path):
            return jsonify({'success': False, 'error': 'Video no encontrado'}), 404
        
        return send_file(video_path, as_attachment=True, download_name='video_anotado.mp4')
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/videos/frame/<path:frame_path>', methods=['GET'])
def get_detection_frame(frame_path):
    """Obtener frame de detección"""
    try:
        full_path = os.path.join(Config.UPLOAD_FOLDER, frame_path)
        if not os.path.exists(full_path):
            return jsonify({'error': 'Frame no encontrado'}), 404
        
        return send_file(full_path, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================
# ENDPOINTS - USUARIOS
# ============================================

@app.route('/api/users', methods=['GET'])
def get_users():
    """Obtener todos los usuarios"""
    try:
        query = "SELECT id, nombre, correo, rol, creado_en FROM usuarios ORDER BY id"
        users = db.execute_query(query, fetch=True)
        return jsonify({'success': True, 'users': users})
    except Exception as e:
        logger.error(f"Error obteniendo usuarios: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/users', methods=['POST'])
def add_user():
    """Agregar nuevo usuario"""
    try:
        data = request.json
        
        required = ['nombre', 'correo', 'password', 'rol']
        if not all(k in data for k in required):
            return jsonify({'success': False, 'error': 'Faltan campos requeridos'}), 400
        
        import hashlib
        password_hash = hashlib.sha256(data['password'].encode()).hexdigest()
        
        query = """
            INSERT INTO usuarios (nombre, correo, password, rol)
            VALUES (%s, %s, %s, %s)
        """
        user_id = db.execute_query(query, (data['nombre'], data['correo'], password_hash, data['rol']))
        
        logger.info(f"✓ Usuario agregado - ID: {user_id}")
        return jsonify({'success': True, 'user_id': user_id, 'message': 'Usuario agregado'})
    
    except Exception as e:
        logger.error(f"Error agregando usuario: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    """Eliminar usuario"""
    try:
        query = "DELETE FROM usuarios WHERE id = %s"
        db.execute_query(query, (user_id,))
        return jsonify({'success': True, 'message': 'Usuario eliminado'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================
# ENDPOINT PRINCIPAL
# ============================================

@app.route('/')
def index():
    """Servir frontend"""
    index_path = os.path.join(FRONTEND_DIR, 'index.html')
    if os.path.exists(index_path):
        return send_file(index_path)
    return jsonify({'error': 'index.html no encontrado', 'path_checked': index_path}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check del sistema"""
    try:
        # Verificar BD
        db.execute_query("SELECT 1", fetch=True)
        
        return jsonify({
            'status': 'healthy',
            'database': 'connected',
            'active_streams': len(camera_manager.get_active_cameras()),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    try:
        logger.info(f"🚀 Servidor iniciando en {Config.HOST}:{Config.PORT}")
        logger.info(f"📁 Frontend: {FRONTEND_DIR}")
        logger.info(f"🎥 YOLO: {'Activado' if camera_manager.use_yolo else 'Desactivado'}")
        logger.info("=" * 60)
        
        socketio.run(
            app,
            host=Config.HOST,
            port=Config.PORT,
            debug=Config.DEBUG,
            use_reloader=False  # Importante: desactivar reloader con threading
        )
    except KeyboardInterrupt:
        logger.info("\n🛑 Deteniendo servidor...")
        camera_manager.stop_all()
        logger.info("✅ Servidor detenido correctamente")
    except Exception as e:
        logger.error(f"💥 Error fatal: {e}")
        camera_manager.stop_all()