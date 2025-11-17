from flask import Flask, request, jsonify, send_file, send_from_directory, Response, make_response
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from functools import wraps
import logging
import os
import cv2
import jwt
import hashlib
from datetime import datetime, timedelta
from config import Config
from database import db
from services.camera_service import CameraManager
from services.video_service import VideoService

FRONTEND_DIR = r"C:\Users\Ramirez\Desktop\ACCIDENT\FRONTED"

# Inicializar Flask
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')

# Configuración
app.config['SECRET_KEY'] = 'accident-detection-secret-key-2025'
app.config['JWT_SECRET_KEY'] = 'jwt-secret-key-super-secure-2025'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# CORS
CORS(app, resources={r"/*": {"origins": "*"}})

# Socket.IO
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    logger=True,
    engineio_logger=False,
    ping_timeout=60,
    ping_interval=25,
    max_http_buffer_size=10000000
)

# Inicializar servicios
Config.init_folders()
camera_manager = CameraManager(socketio, use_yolo=True)

logger.info("=" * 60)
logger.info("🚨 SISTEMA DE DETECCIÓN DE ACCIDENTES INICIADO")
logger.info("=" * 60)

# ============================================
# PERMISOS POR ROL
# ============================================

ROLE_PERMISSIONS = {
    'admin': {
        'dashboard': True,
        'camaras': True,
        'agregar_camara': True,
        'monitoreo': True,
        'registros': True,
        'subir_video': True,
        'estadisticas': True,
        'mapa': True,
        'config': True,
        'usuarios': True,
        'reportes': True,
        'eliminar_camara': True,
        'iniciar_camara': True,
        'detener_camara': True
    },
    'operador': {
        'dashboard': True,
        'camaras': True,
        'agregar_camara': False,
        'monitoreo': True,
        'registros': True,
        'subir_video': False,
        'estadisticas': True,
        'mapa': True,
        'config': False,
        'usuarios': False,
        'reportes': False,
        'eliminar_camara': False,
        'iniciar_camara': True,
        'detener_camara': True
    },
    'emergencias': {
        'dashboard': True,
        'camaras': False,
        'agregar_camara': False,
        'monitoreo': False,
        'registros': False,
        'subir_video': False,
        'estadisticas': False,
        'mapa': False,
        'config': False,
        'usuarios': False,
        'reportes': False,
        'eliminar_camara': False,
        'iniciar_camara': False,
        'detener_camara': False
    }
}

# ============================================
# DECORADORES DE AUTENTICACIÓN
# ============================================

def token_required(f):
    """Verificar que el usuario tiene un token válido"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Token en header Authorization
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(" ")[1]  # "Bearer TOKEN"
            except IndexError:
                return jsonify({'success': False, 'error': 'Token inválido'}), 401
        
        # Token en query params (para debugging)
        if not token and 'token' in request.args:
            token = request.args.get('token')
        
        if not token:
            return jsonify({'success': False, 'error': 'Token no proporcionado'}), 401
        
        try:
            # Decodificar token
            data = jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=["HS256"])
            current_user = db.get_user_by_id(data['user_id'])
            
            if not current_user:
                return jsonify({'success': False, 'error': 'Usuario no encontrado'}), 401
            
            # Verificar que el usuario esté activo
            if not current_user.get('activo', True):
                return jsonify({'success': False, 'error': 'Usuario inactivo'}), 401
            
        except jwt.ExpiredSignatureError:
            return jsonify({'success': False, 'error': 'Token expirado'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'success': False, 'error': 'Token inválido'}), 401
        
        return f(current_user, *args, **kwargs)
    
    return decorated

def role_required(*allowed_roles):
    """Verificar que el usuario tiene uno de los roles permitidos"""
    def decorator(f):
        @wraps(f)
        @token_required
        def decorated(current_user, *args, **kwargs):
            if current_user['role'] not in allowed_roles:
                logger.warning(f"⚠️ Acceso denegado - Usuario: {current_user['nombre']}, Rol: {current_user['role']}")
                return jsonify({
                    'success': False, 
                    'error': 'Permisos insuficientes',
                    'required_roles': list(allowed_roles),
                    'your_role': current_user['role']
                }), 403
            
            return f(current_user, *args, **kwargs)
        
        return decorated
    return decorator

def permission_required(permission):
    """Verificar que el usuario tiene un permiso específico"""
    def decorator(f):
        @wraps(f)
        @token_required
        def decorated(current_user, *args, **kwargs):
            user_role = current_user['role']
            
            if user_role not in ROLE_PERMISSIONS:
                return jsonify({'success': False, 'error': 'Rol no reconocido'}), 403
            
            if not ROLE_PERMISSIONS[user_role].get(permission, False):
                logger.warning(f"⚠️ Permiso denegado - Usuario: {current_user['nombre']}, Permiso: {permission}")
                return jsonify({
                    'success': False,
                    'error': f'No tienes permiso para: {permission}',
                    'your_role': user_role
                }), 403
            
            return f(current_user, *args, **kwargs)
        
        return decorated
    return decorator

# ============================================
# ENDPOINTS - AUTENTICACIÓN
# ============================================

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login de usuario"""
    try:
        data = request.json
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'success': False, 'error': 'Usuario y contraseña requeridos'}), 400
        
        # Buscar usuario (puede ser nombre o correo)
        user = db.get_user_by_nombre(username)
        if not user:
            user = db.get_user_by_correo(username)
        
        if not user:
            logger.warning(f"⚠️ Intento de login fallido - Usuario no existe: {username}")
            return jsonify({'success': False, 'error': 'Usuario o contraseña incorrectos'}), 401
        
        # Verificar si está bloqueado
        if user.get('bloqueado_hasta') and datetime.now() < user['bloqueado_hasta']:
            return jsonify({
                'success': False, 
                'error': 'Usuario bloqueado temporalmente por múltiples intentos fallidos'
            }), 403
        
        # Verificar contraseña
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        if user['password'] != password_hash:
            # Incrementar intentos fallidos
            db.increment_failed_attempts(user['id'])
            logger.warning(f"⚠️ Contraseña incorrecta - Usuario: {username}")
            return jsonify({'success': False, 'error': 'Usuario o contraseña incorrectos'}), 401
        
        # Login exitoso
        db.reset_failed_attempts(user['id'])
        db.update_last_login(user['id'])
        
        # Generar token JWT
        token = jwt.encode({
            'user_id': user['id'],
            'role': user['role'],
            'exp': datetime.utcnow() + timedelta(hours=24)
        }, app.config['JWT_SECRET_KEY'], algorithm="HS256")
        
        # Obtener permisos del usuario
        permissions = ROLE_PERMISSIONS.get(user['role'], {})
        
        logger.info(f"✅ Login exitoso - Usuario: {user['nombre']}, Rol: {user['role']}")
        
        # Registrar en logs de auditoría
        db.log_action(
            user_id=user['id'],
            action='LOGIN',
            detalle=f"Login exitoso desde {request.remote_addr}",
            ip_address=request.remote_addr,
            status='SUCCESS'
        )
        
        return jsonify({
            'success': True,
            'token': token,
            'user': {
                'id': user['id'],
                'nombre': user['nombre'],
                'correo': user['correo'],
                'rol': user['role']
            },
            'permissions': permissions
        })
    
    except Exception as e:
        logger.error(f"Error en login: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/auth/verify', methods=['GET'])
@token_required
def verify_token(current_user):
    """Verificar si el token es válido"""
    permissions = ROLE_PERMISSIONS.get(current_user['role'], {})
    
    return jsonify({
        'success': True,
        'user': {
            'id': current_user['id'],
            'nombre': current_user['nombre'],
            'correo': current_user['correo'],
            'rol': current_user['role']
        },
        'permissions': permissions
    })

@app.route('/api/auth/logout', methods=['POST'])
@token_required
def logout(current_user):
    """Logout de usuario"""
    db.log_action(
        user_id=current_user['id'],
        action='LOGOUT',
        detalle=f"Logout desde {request.remote_addr}",
        ip_address=request.remote_addr
    )
    
    return jsonify({'success': True, 'message': 'Sesión cerrada'})

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

# ============================================
# ENDPOINTS - CÁMARAS (CON PERMISOS)
# ============================================

@app.route('/api/cameras', methods=['GET'])
@token_required
def get_cameras(current_user):
    """Obtener todas las cámaras"""
    try:
        cameras = db.get_all_cameras()
        active = camera_manager.get_active_cameras()
        
        for cam in cameras:
            cam['is_streaming'] = cam['id'] in active
        
        logger.info(f"📹 Usuario {current_user['nombre']} consultó {len(cameras)} cámaras")
        return jsonify({'success': True, 'cameras': cameras})
    except Exception as e:
        logger.error(f"Error obteniendo cámaras: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/cameras', methods=['POST'])
@permission_required('agregar_camara')
def add_camera(current_user):
    """Agregar nueva cámara (solo admin)"""
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
        
        # Registrar en auditoría
        db.log_action(
            user_id=current_user['id'],
            action='ADD_CAMERA',
            table_affected='camaras',
            record_id=camera_id,
            new_value=f"IP: {data['ip']}, Puerto: {data['puerto']}",
            ip_address=request.remote_addr
        )
        
        logger.info(f"✅ Cámara agregada por {current_user['nombre']} - ID: {camera_id}")
        return jsonify({'success': True, 'camera_id': camera_id})
    
    except Exception as e:
        logger.error(f"Error agregando cámara: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/cameras/<int:camera_id>', methods=['DELETE'])
@permission_required('eliminar_camara')
def delete_camera(current_user, camera_id):
    """Eliminar cámara (solo admin)"""
    try:
        camera = db.get_camera_by_id(camera_id)
        if not camera:
            return jsonify({'success': False, 'error': 'Cámara no encontrada'}), 404
        
        if camera_manager.is_camera_active(camera_id):
            camera_manager.stop_camera(camera_id)
        
        db.delete_camera(camera_id)
        
        db.log_action(
            user_id=current_user['id'],
            action='DELETE_CAMERA',
            table_affected='camaras',
            record_id=camera_id,
            old_value=f"IP: {camera['ip']}",
            ip_address=request.remote_addr
        )
        
        logger.info(f"🗑️ Cámara {camera_id} eliminada por {current_user['nombre']}")
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/cameras/<int:camera_id>/start', methods=['POST'])
@permission_required('iniciar_camara')
def start_camera_stream(current_user, camera_id):
    """Iniciar stream (admin y operador)"""
    try:
        success = camera_manager.start_camera(camera_id)
        
        if success:
            db.log_action(
                user_id=current_user['id'],
                action='START_CAMERA',
                table_affected='camaras',
                record_id=camera_id,
                ip_address=request.remote_addr
            )
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'No se pudo iniciar'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/cameras/<int:camera_id>/stop', methods=['POST'])
@permission_required('detener_camara')
def stop_camera_stream(current_user, camera_id):
    """Detener stream (admin y operado  r)"""
    try:
        success = camera_manager.stop_camera(camera_id)
        
        if success:
            db.log_action(
                user_id=current_user['id'],
                action='STOP_CAMERA',
                table_affected='camaras',
                record_id=camera_id,
                ip_address=request.remote_addr
            )
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Stream no activo'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================
# ENDPOINTS - ACCIDENTES (CON PERMISOS)
# ============================================

@app.route('/api/accidents', methods=['GET'])
@token_required
def get_accidents(current_user):
    """Obtener accidentes (todos los roles)"""
    try:
        limit = request.args.get('limit', 100, type=int)
        accidents = db.get_all_accidents(limit)
        return jsonify({'success': True, 'accidents': accidents})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/accidents/export', methods=['GET'])
@token_required
def export_accidents_csv(current_user):
    """Exportar accidentes a CSV"""
    try:
        import csv
        import io
        
        accidents = db.get_all_accidents(1000)
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['ID', 'Cámara', 'Fecha', 'Latitud', 'Longitud', 'Descripción'])
        
        for acc in accidents:
            writer.writerow([
                acc['id'],
                acc.get('camera_ip', 'N/A'),
                acc['fecha_accidente'],
                acc.get('latitud', ''),
                acc.get('longitud', ''),
                acc.get('descripcion', '')
            ])
        
        output.seek(0)
        
        db.log_action(
            user_id=current_user['id'],
            action='EXPORT_CSV',
            detalle=f"Exportó {len(accidents)} accidentes",
            ip_address=request.remote_addr
        )
        
        return output.getvalue(), 200, {
            'Content-Type': 'text/csv',
            'Content-Disposition': f'attachment; filename=accidentes_{datetime.now().strftime("%Y%m%d")}.csv'
        }
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================
# ENDPOINTS - USUARIOS (SOLO ADMIN)
# ============================================

@app.route('/api/users', methods=['GET'])
@role_required('admin')
def get_users(current_user):
    """Obtener usuarios (solo admin)"""
    try:
        users = db.get_all_users()
        return jsonify({'success': True, 'users': users})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/users', methods=['POST'])
@role_required('admin')
def add_user(current_user):
    """Agregar usuario (solo admin)"""
    try:
        data = request.json
        
        password_hash = hashlib.sha256(data['password'].encode()).hexdigest()
        
        user_id = db.create_user(
            nombre=data['nombre'],
            correo=data['correo'],
            password_hash=password_hash,
            role=data['rol']
        )
        
        db.log_action(
            user_id=current_user['id'],
            action='CREATE_USER',
            table_affected='usuarios',
            record_id=user_id,
            new_value=f"Usuario: {data['nombre']}, Rol: {data['rol']}",
            ip_address=request.remote_addr
        )
        
        return jsonify({'success': True, 'user_id': user_id})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/users/<int:user_id>', methods=['DELETE'])
@role_required('admin')
def delete_user(current_user, user_id):
    """Eliminar usuario (solo admin)"""
    try:
        query = "DELETE FROM usuarios WHERE id = %s"
        db.execute_query(query, (user_id,))
        
        db.log_action(
            user_id=current_user['id'],
            action='DELETE_USER',
            table_affected='usuarios',
            record_id=user_id,
            ip_address=request.remote_addr
        )
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


        # Cambios
@app.route('/api/users/<int:user_id>', methods=['PUT'])
@role_required('admin')
def update_user(current_user, user_id):
    """Actualizar usuario (solo admin)"""
    try:
        data = request.json
        user = db.get_user_by_id(user_id)
        
        if not user:
            return jsonify({'success': False, 'error': 'Usuario no encontrado'}), 404
        
        # Construir query dinámico
        updates = []
        params = []
        
        if 'nombre' in data:
            updates.append("nombre = %s")
            params.append(data['nombre'])
        
        if 'correo' in data:
            updates.append("correo = %s")
            params.append(data['correo'])
        
        if 'password' in data and data['password']:
            password_hash = hashlib.sha256(data['password'].encode()).hexdigest()
            updates.append("password = %s")
            params.append(password_hash)
        
        if 'rol' in data:
            updates.append("role = %s")
            params.append(data['rol'])
        
        if not updates:
            return jsonify({'success': False, 'error': 'No hay campos para actualizar'}), 400
        
        params.append(user_id)
        query = f"UPDATE usuarios SET {', '.join(updates)} WHERE id = %s"
        
        db.execute_query(query, tuple(params))
        
        db.log_action(
            user_id=current_user['id'],
            action='UPDATE_USER',
            table_affected='usuarios',
            record_id=user_id,
            old_value=f"Nombre: {user['nombre']}",
            new_value=f"Nombre: {data.get('nombre', user['nombre'])}",
            ip_address=request.remote_addr
        )
        
        logger.info(f"✅ Usuario {user_id} actualizado por {current_user['nombre']}")
        return jsonify({'success': True, 'message': 'Usuario actualizado'})
    except Exception as e:
        logger.error(f"Error actualizando usuario: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
# ============================================
# ENDPOINTS - VIDEOS (ADMIN Y ANALISTA)
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
# ENDPOINTS - CONFIGURACIÓN DEL SISTEMA
# ============================================

@app.route('/api/config', methods=['GET'])
@token_required
def get_system_config(current_user):
    """Obtener configuración del sistema"""
    try:
        config = db.get_all_config()
        return jsonify({'success': True, 'config': config})
    except Exception as e:
        logger.error(f"Error obteniendo configuración: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/config', methods=['POST'])
@role_required('admin')
def update_system_config(current_user):
    """Actualizar configuración (solo admin)"""
    try:
        data = request.json
        
        if not data:
            return jsonify({'success': False, 'error': 'No se enviaron datos'}), 400
        
        # Actualizar cada configuración
        updated = []
        errors = []
        
        for clave, valor in data.items():
            if db.set_config(clave, valor):
                updated.append(clave)
            else:
                errors.append(clave)
        
        # Registrar en auditoría
        db.log_action(
            user_id=current_user['id'],
            action='UPDATE_CONFIG',
            detalle=f"Configuración actualizada: {', '.join(updated)}",
            ip_address=request.remote_addr
        )
        
        # Reiniciar cámaras activas para aplicar cambios
        active_cameras = camera_manager.get_active_cameras()
        for cam_id in active_cameras:
            camera_manager.stop_camera(cam_id)
            camera_manager.start_camera(cam_id)
        
        logger.info(f"✅ Configuración actualizada por {current_user['nombre']}: {updated}")
        
        return jsonify({
            'success': True,
            'updated': updated,
            'errors': errors,
            'message': 'Configuración actualizada. Cámaras reiniciadas para aplicar cambios.'
        })
    
    except Exception as e:
        logger.error(f"Error actualizando configuración: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    
@app.route('/api/config/live', methods=['GET'])
def get_live_config():
    """Obtener configuración en tiempo real (sin autenticación para updates del frontend)"""
    try:
        config = db.get_all_config()
        
        # Agregar información adicional
        active_cameras = len(camera_manager.get_active_cameras())
        
        return jsonify({
            'success': True,
            'config': config,
            'active_cameras': active_cameras,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error obteniendo configuración live: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    
# ============================================
# ENDPOINTS PÚBLICOS
# ============================================

@app.route('/')
def index():
    """Servir frontend"""
    login_path = os.path.join(FRONTEND_DIR, 'login.html')
    if os.path.exists(login_path):
        return send_file(login_path)
    return jsonify({'error': 'login.html no encontrado'}), 500

@app.route('/index.html')
def dashboard():
    """Servir dashboard (requiere estar logueado)"""
    index_path = os.path.join(FRONTEND_DIR, 'index.html')
    if os.path.exists(index_path):
        return send_file(index_path)
    return jsonify({'error': 'index.html no encontrado'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/videos/<filename>', methods=['GET'])
def serve_video(filename):
    """Servir videos (público para reproducción)"""
    try:
        videos_dir = r"C:\Users\Ramirez\Desktop\ACCIDENT\BACKEND\clips"
        file_path = os.path.join(videos_dir, filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'Video no encontrado'}), 404
        
        return send_file(file_path, mimetype='video/mp4')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cameras/<int:camera_id>/stats', methods=['GET'])
@token_required
def get_camera_stats(current_user, camera_id):
    """Obtener estadísticas de cámara"""
    try:
        stats = camera_manager.get_camera_stats(camera_id)
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500



# ============================================
# ENDPOINTS PARA APP MÓVIL - ALERTAS
# ============================================

@app.route('/api/mobile/image/<int:accident_id>', methods=['GET'])
def get_accident_image_mobile(accident_id):
    """Obtener imagen del accidente para la app móvil"""
    try:
        query = "SELECT ruta_archivo FROM accidentes WHERE id = %s"
        result = db.execute_query(query, (accident_id,), fetch=True)
        
        if not result or not result[0]['ruta_archivo']:
            return jsonify({'success': False, 'error': 'Imagen no disponible'}), 404
        
        video_path = result[0]['ruta_archivo']
        
        # Extraer frame del video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({'success': False, 'error': 'No se pudo abrir video'}), 404
        
        # Tomar frame de la mitad del video
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return jsonify({'success': False, 'error': 'No se pudo extraer frame'}), 404
        
        # Convertir a JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        return Response(buffer.tobytes(), mimetype='image/jpeg')
    
    except Exception as e:
        logger.error(f"Error obteniendo imagen: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================
# WEBSOCKET PARA APP MÓVIL
# ============================================

@socketio.on('mobile_connect')
def handle_mobile_connect(data):
    """App móvil se conecta para recibir alertas"""
    from flask_socketio import join_room
    
    user_id = data.get('user_id', 'anonymous')
    logger.info(f"📱 App móvil conectada - Usuario: {user_id}, SID: {request.sid}")
    
    # Unir a room de alertas de emergencia
    join_room('mobile_emergency')
    
    emit('mobile_connected', {
        'status': 'connected',
        'message': 'Conectado - Recibirás alertas en tiempo real',
        'timestamp': datetime.now().isoformat()
    })


# ============================================
# ENDPOINTS PARA REPORTES
# ============================================
@app.route('/api/reports/audit', methods=['GET'])
@token_required
def get_audit_reports(current_user):
    """Obtener reportes de auditoría desde la tabla logs"""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        query = """
            SELECT 
                l.id,
                l.action AS tipo_evento,
                l.detalle AS descripcion,
                l.creado_en AS fecha_hora,
                l.ip_address AS ip_usuario,
                u.nombre AS nombre_usuario,
                u.role AS rol
            FROM logs l
            LEFT JOIN usuarios u ON l.id_usuario = u.id
            WHERE 1=1
        """
        params = []

        if start_date:
            query += " AND DATE(l.creado_en) >= %s"
            params.append(start_date)

        if end_date:
            query += " AND DATE(l.creado_en) <= %s"
            params.append(end_date)

        query += " ORDER BY l.creado_en DESC LIMIT 1000"

        results = db.execute_query(query, tuple(params), fetch=True)

        return jsonify({
            "success": True,
            "data": results or [],
            "count": len(results) if results else 0
        })

    except Exception as e:
        logger.error(f"Error obteniendo reportes de auditoría: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/reports/access', methods=['GET'])
@token_required
def get_access_reports(current_user):
    """Obtener reportes de accesos al sistema usando tabla logs"""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        query = """
            SELECT
                l.id,
                l.creado_en AS fecha_hora,
                l.ip_address AS ip_usuario,
                l.detalle AS descripcion,
                u.nombre AS nombre_usuario,
                u.correo,
                u.role AS rol,
                CASE
                    WHEN l.action = 'LOGIN' AND l.status = 'SUCCESS' THEN 'Exitoso'
                    WHEN l.action = 'LOGIN' AND l.status != 'SUCCESS' THEN 'Fallido'
                    WHEN l.action = 'LOGOUT' THEN 'Cierre de sesión'
                    ELSE 'Otro'
                END AS estado
            FROM logs l
            LEFT JOIN usuarios u ON l.id_usuario = u.id
            WHERE l.action IN ('LOGIN', 'LOGOUT')
        """

        params = []

        if start_date:
            query += " AND DATE(l.creado_en) >= %s"
            params.append(start_date)

        if end_date:
            query += " AND DATE(l.creado_en) <= %s"
            params.append(end_date)

        query += " ORDER BY l.creado_en DESC LIMIT 1000"

        results = db.execute_query(query, tuple(params), fetch=True)

        return jsonify({
            "success": True,
            "data": results or [],
            "count": len(results) if results else 0
        })

    except Exception as e:
        logger.error(f"Error obteniendo reportes de acceso: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/reports/accidents', methods=['GET'])
@token_required
def get_accident_reports(current_user):
    """Obtener reportes de accidentes"""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        camera_id = request.args.get('camera_id')

        query = """
            SELECT 
                a.id,
                a.fecha_accidente AS fecha_deteccion,
                a.latitud,
                a.longitud,
                a.descripcion,
                a.severidad,
                a.estado,
                a.ruta_archivo,
                c.ip AS camera_ip,
                c.puerto AS camera_port,
                c.ciudad AS camera_ciudad,
                c.direccion AS camera_direccion,
                c.zona AS camera_zona
            FROM accidentes a
            LEFT JOIN camaras c ON a.id_camara = c.id
            WHERE 1=1
        """

        params = []

        if start_date:
            query += " AND DATE(a.fecha_accidente) >= %s"
            params.append(start_date)

        if end_date:
            query += " AND DATE(a.fecha_accidente) <= %s"
            params.append(end_date)

        if camera_id:
            query += " AND a.id_camara = %s"
            params.append(camera_id)

        query += " ORDER BY a.fecha_accidente DESC LIMIT 500"

        results = db.execute_query(query, tuple(params), fetch=True)

        return jsonify({
            "success": True,
            "data": results or [],
            "count": len(results) if results else 0
        })

    except Exception as e:
        logger.error(f"Error obteniendo reportes de accidentes: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    try:
        logger.info(f"🚀 Servidor iniciando en {Config.HOST}:{Config.PORT}")
        logger.info(f"🔐 Sistema de autenticación: ACTIVADO")
        logger.info("=" * 60)
        
        socketio.run(
            app,
            host=Config.HOST,
            port=Config.PORT,
            debug=Config.DEBUG,
            use_reloader=False
        )
    except KeyboardInterrupt:
        logger.info("\n🛑 Deteniendo servidor...")
        camera_manager.stop_all()
    except Exception as e:
        logger.error(f"💥 Error fatal: {e}")
        camera_manager.stop_all()