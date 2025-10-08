import mysql.connector
from mysql.connector import Error, pooling
from config import Config
from contextlib import contextmanager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Database:
    _instance = None
    _pool = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Database, cls).__new__(cls)
            cls._instance._initialize_pool()
        return cls._instance
    
    def _initialize_pool(self):
        """Inicializar pool de conexiones"""
        try:
            self._pool = pooling.MySQLConnectionPool(
                pool_name="accident_pool",
                pool_size=5,
                pool_reset_session=True,
                host=Config.DB_HOST,
                user=Config.DB_USER,
                password=Config.DB_PASSWORD,
                database=Config.DB_NAME
            )
            logger.info("✓ Pool de conexiones MySQL inicializado")
        except Error as e:
            logger.error(f"✗ Error al crear pool de conexiones: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Context manager para conexiones del pool"""
        conn = None
        try:
            conn = self._pool.get_connection()
            yield conn
        except Error as e:
            logger.error(f"Error de base de datos: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn and conn.is_connected():
                conn.close()
    
    def execute_query(self, query, params=None, fetch=False):
        """Ejecutar query con manejo de errores"""
        with self.get_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            try:
                cursor.execute(query, params or ())
                
                if fetch:
                    result = cursor.fetchall()
                    return result
                else:
                    conn.commit()
                    return cursor.lastrowid
            except Error as e:
                logger.error(f"Error en query: {e}\nQuery: {query}\nParams: {params}")
                raise
            finally:
                cursor.close()
    
    def execute_many(self, query, data):
        """Insertar múltiples registros"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.executemany(query, data)
                conn.commit()
                return cursor.rowcount
            except Error as e:
                logger.error(f"Error en executemany: {e}")
                raise
            finally:
                cursor.close()
    
    # ===== MÉTODOS ESPECÍFICOS PARA TUS TABLAS =====
    
    def get_all_cameras(self):
        """Obtener todas las cámaras"""
        query = "SELECT * FROM camaras ORDER BY id"
        return self.execute_query(query, fetch=True)
    
    def get_camera_by_id(self, camera_id):
        """Obtener una cámara por ID"""
        query = "SELECT * FROM camaras WHERE id = %s"
        result = self.execute_query(query, (camera_id,), fetch=True)
        return result[0] if result else None
    
    def add_camera(self, ip, puerto, usuario, password, latitud, longitud, url_rtsp):
        """Agregar nueva cámara"""
        query = """
            INSERT INTO camaras (ip, puerto, usuario, password, latitud, longitud, url_rtsp)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        return self.execute_query(query, (ip, puerto, usuario, password, latitud, longitud, url_rtsp))
    
    def update_camera(self, camera_id, ip, puerto, usuario, password, latitud, longitud, url_rtsp):
        """Actualizar cámara"""
        query = """
            UPDATE camaras 
            SET ip=%s, puerto=%s, usuario=%s, password=%s, latitud=%s, longitud=%s, url_rtsp=%s
            WHERE id=%s
        """
        self.execute_query(query, (ip, puerto, usuario, password, latitud, longitud, url_rtsp, camera_id))
    
    def delete_camera(self, camera_id):
        """Eliminar cámara"""
        query = "DELETE FROM camaras WHERE id = %s"
        self.execute_query(query, (camera_id,))
    
    def save_accident(self, id_camara, ruta_archivo, latitud, longitud, descripcion):
        """Guardar accidente detectado"""
        query = """
            INSERT INTO accidentes (id_camara, ruta_archivo, latitud, longitud, descripcion)
            VALUES (%s, %s, %s, %s, %s)
        """
        return self.execute_query(query, (id_camara, ruta_archivo, latitud, longitud, descripcion))
    
    def save_evidence(self, id_accidente, tipo, ruta_archivo, anonimizado=0):
        """Guardar evidencia (imagen/video)"""
        query = """
            INSERT INTO evidencias (id_accidente, tipo, ruta_archivo, anonimizado)
            VALUES (%s, %s, %s, %s)
        """
        return self.execute_query(query, (id_accidente, tipo, ruta_archivo, anonimizado))
    
    def get_accidents_by_camera(self, camera_id, limit=100):
        """Obtener accidentes de una cámara"""
        query = """
            SELECT a.*, c.ip as camera_ip, c.latitud as cam_lat, c.longitud as cam_lng
            FROM accidentes a
            JOIN camaras c ON a.id_camara = c.id
            WHERE a.id_camara = %s
            ORDER BY a.fecha_accidente DESC
            LIMIT %s
        """
        return self.execute_query(query, (camera_id, limit), fetch=True)
    
    def get_all_accidents(self, limit=100):
        """Obtener todos los accidentes"""
        query = """
            SELECT a.*, c.ip as camera_ip
            FROM accidentes a
            JOIN camaras c ON a.id_camara = c.id
            ORDER BY a.fecha_accidente DESC
            LIMIT %s
        """
        return self.execute_query(query, (limit,), fetch=True)
    
    def log_action(self, id_usuario, accion, detalle):
        """Registrar log de acción"""
        query = """
            INSERT INTO logs (id_usuario, accion, detalle)
            VALUES (%s, %s, %s)
        """
        self.execute_query(query, (id_usuario, accion, detalle))

# Instancia global singleton
db = Database()