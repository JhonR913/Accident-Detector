import mysql.connector
from mysql.connector import Error, pooling
from config import Config
from contextlib import contextmanager
import logging
from datetime import datetime

# Nueva importación para cifrado
try:
    from cryptography.fernet import Fernet
except Exception:
    Fernet = None

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

    # ------------------ Helpers de cifrado ------------------
    def _get_fernet(self):
        """
        Devuelve una instancia Fernet si Config tiene clave válida.
        Busca Config.FERNET_KEY, Config.ENCRYPTION_KEY o Config.DB_ENCRYPTION_KEY.
        """
        key = getattr(Config, "FERNET_KEY", None) \
              or getattr(Config, "ENCRYPTION_KEY", None) \
              or getattr(Config, "DB_ENCRYPTION_KEY", None)
        if not key:
            logger.debug("🔒 No se encontró clave Fernet en Config")
            return None
        if Fernet is None:
            logger.warning("⚠️ cryptography.Fernet no está disponible (instala cryptography).")
            return None
        try:
            if isinstance(key, str):
                key = key.encode()
            return Fernet(key)
        except Exception as e:
            logger.error(f"✗ Error creando Fernet desde Config key: {e}")
            return None

    def _encrypt_credential(self, value):
        """Cifra un valor con Fernet y devuelve string. Si no puede, devuelve el valor crudo."""
        if value is None:
            return None
        f = self._get_fernet()
        if not f:
            logger.debug("🔒 No hay Fernet: guardando credencial en texto plano (fallback).")
            return value
        try:
            token = f.encrypt(str(value).encode())
            return token.decode()
        except Exception as e:
            logger.error(f"✗ Error cifrando credencial: {e}. Guardando crudo.")
            return value

    # ==================== MÉTODOS DE CÁMARAS ====================

    def get_all_cameras(self):
        """Obtener todas las cámaras (sin mostrar credenciales)"""
        query = """
            SELECT id, ip, puerto, latitud, longitud,
                   ciudad, direccion, zona,
                   privacy_notice_enabled, normative_compliance_verified,
                   encryption_status, creado_en
            FROM camaras 
            ORDER BY id
        """
        return self.execute_query(query, fetch=True)

    def get_camera_by_id(self, camera_id):
        """Obtener cámara por ID (incluye credenciales cifradas y URL cifrada si existe)"""
        query = """
            SELECT id, ip, puerto, latitud, longitud,
                   url_rtsp_cifrada, ciudad, direccion, zona,
                   usuario_cifrado, password_cifrado, encryption_status,
                   privacy_notice_enabled, location_full_description
            FROM camaras 
            WHERE id = %s
        """
        result = self.execute_query(query, (camera_id,), fetch=True)
        return result[0] if result else None

    def add_camera(self, ip, puerto, latitud, longitud, url_rtsp=None, ciudad=None, 
                   direccion=None, zona='CENTRO', usuario=None, password=None,
                   usuario_cifrado=None, password_cifrado=None,
                   location_full_description=None):
        """
        Agregar nueva cámara.
        Cifra automáticamente usuario, password y URL RTSP.
        """
        # Preferir valores ya cifrados; si no existen, cifrar los plaintext
        u_cif = usuario_cifrado or (self._encrypt_credential(usuario) if usuario else None)
        p_cif = password_cifrado or (self._encrypt_credential(password) if password else None)

        # Construir URL RTSP solo si hay usuario y password
        if usuario and password:
            rtsp_url = f"rtsp://{usuario}:{password}@{ip}:{puerto}/Streaming/Channels/101"
        else:
            rtsp_url = f"rtsp://{ip}:{puerto}/Streaming/Channels/101"

        # Cifrar la URL
        url_rtsp_cifrada = self._encrypt_credential(rtsp_url)

        query = """
            INSERT INTO camaras (
                ip, puerto, latitud, longitud, url_rtsp_cifrada, ciudad, direccion, zona,
                usuario_cifrado, password_cifrado, encryption_status,
                location_full_description, normative_compliance_verified
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'ENCRYPTED', %s, TRUE)
        """
        return self.execute_query(query, (
            ip, puerto, latitud, longitud, url_rtsp_cifrada,
            ciudad, direccion, zona, u_cif, p_cif, location_full_description
        ))

    def update_camera(self, camera_id, ip, puerto, latitud, longitud, url_rtsp=None, 
                      ciudad=None, direccion=None, zona=None, usuario=None, password=None,
                      usuario_cifrado=None, password_cifrado=None):
        """
        Actualizar cámara.
        Actualiza credenciales y URL RTSP cifrada.
        """
        u_cif = usuario_cifrado or (self._encrypt_credential(usuario) if usuario else None)
        p_cif = password_cifrado or (self._encrypt_credential(password) if password else None)

        if usuario and password:
            rtsp_url = f"rtsp://{usuario}:{password}@{ip}:{puerto}/Streaming/Channels/101"
        else:
            rtsp_url = f"rtsp://{ip}:{puerto}/Streaming/Channels/101"

        url_rtsp_cifrada = self._encrypt_credential(rtsp_url)

        query = """
            UPDATE camaras 
            SET ip=%s, puerto=%s, latitud=%s, longitud=%s, url_rtsp_cifrada=%s, 
                ciudad=%s, direccion=%s, zona=%s, usuario_cifrado=%s, password_cifrado=%s
            WHERE id=%s
        """
        self.execute_query(query, (
            ip, puerto, latitud, longitud, url_rtsp_cifrada,
            ciudad, direccion, zona, u_cif, p_cif, camera_id
        ))

    def delete_camera(self, camera_id):
        """Eliminar cámara (el trigger auditará automáticamente)"""
        query = "DELETE FROM camaras WHERE id = %s"
        self.execute_query(query, (camera_id,))


    def get_cameras_pending_encryption(self):
        """Obtener cámaras con credenciales sin cifrar"""
        query = """
            SELECT id, ip, usuario, password
            FROM camaras 
            WHERE encryption_status = 'PENDING' 
            AND usuario IS NOT NULL
        """
        return self.execute_query(query, fetch=True)

    def update_camera_credentials(self, camera_id, usuario_cifrado, password_cifrado):
        """Actualizar credenciales cifradas y limpiar texto plano"""
        query = """
            UPDATE camaras 
            SET usuario_cifrado = %s, 
                password_cifrado = %s,
                usuario = NULL,
                password = NULL,
                encryption_status = 'ENCRYPTED'
            WHERE id = %s
        """
        self.execute_query(query, (usuario_cifrado, password_cifrado, camera_id))

    # ==================== MÉTODOS DE ACCIDENTES ====================

    def save_accident(self, id_camara, ruta_archivo, latitud, longitud, descripcion, 
                     severidad='MEDIA', ciudad=None, direccion=None, 
                     vehiculos_involucrados=0, estado='ACTIVO'):
        """Guardar accidente detectado (trigger auditará automáticamente)"""
        query = """
            INSERT INTO accidentes (id_camara, ruta_archivo, latitud, longitud, 
                                   descripcion, severidad, ciudad, direccion,
                                   vehiculos_involucrados, estado)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        return self.execute_query(query, (id_camara, ruta_archivo, latitud, longitud, 
                                         descripcion, severidad, ciudad, direccion,
                                         vehiculos_involucrados, estado))

    def update_accident_status(self, accident_id, estado, descripcion_adicional=None):
        """Actualizar estado de accidente"""
        if descripcion_adicional:
            query = """
                UPDATE accidentes 
                SET estado = %s, 
                    descripcion = CONCAT(descripcion, '\n\nActualización: ', %s)
                WHERE id = %s
            """
            self.execute_query(query, (estado, descripcion_adicional, accident_id))
        else:
            query = "UPDATE accidentes SET estado = %s WHERE id = %s"
            self.execute_query(query, (estado, accident_id))

    def save_evidence(self, id_accidente, tipo, ruta_archivo, anonimizado=0):
        """Guardar evidencia (imagen/video)"""
        query = """
            INSERT INTO evidencias (id_accidente, tipo, ruta_archivo, anonimizado)
            VALUES (%s, %s, %s, %s)
        """
        return self.execute_query(query, (id_accidente, tipo, ruta_archivo, anonimizado))

    def get_accidents_by_camera(self, camera_id, limit=100):
        """Obtener accidentes de una cámara específica"""
        query = """
            SELECT a.*, c.ip as camera_ip, c.latitud as cam_lat, c.longitud as cam_lng,
                   c.ciudad as camera_ciudad,
                   (SELECT COUNT(*) FROM evidencias e WHERE e.id_accidente = a.id) as total_evidencias
            FROM accidentes a
            JOIN camaras c ON a.id_camara = c.id
            WHERE a.id_camara = %s
            ORDER BY a.fecha_accidente DESC
            LIMIT %s
        """
        return self.execute_query(query, (camera_id, limit), fetch=True)

    def get_all_accidents(self, limit=100, date_from=None, date_to=None, 
                         camera_id=None, ciudad=None, severidad=None, estado=None):
        """Obtener accidentes con filtros opcionales"""
        query = """
            SELECT a.*, 
                   c.ip as camera_ip, 
                   c.ciudad as camera_ciudad,
                   c.direccion as camera_direccion,
                   (SELECT COUNT(*) FROM evidencias e WHERE e.id_accidente = a.id) as total_evidencias
            FROM accidentes a
            JOIN camaras c ON a.id_camara = c.id
            WHERE 1=1
        """
        params = []

        if date_from:
            query += " AND a.fecha_accidente >= %s"
            params.append(date_from)

        if date_to:
            query += " AND a.fecha_accidente <= %s"
            params.append(date_to)

        if camera_id:
            query += " AND a.id_camara = %s"
            params.append(camera_id)

        if ciudad:
            query += " AND (a.ciudad = %s OR c.ciudad = %s)"
            params.extend([ciudad, ciudad])

        if severidad:
            query += " AND a.severidad = %s"
            params.append(severidad)

        if estado:
            query += " AND a.estado = %s"
            params.append(estado)

        query += " ORDER BY a.fecha_accidente DESC LIMIT %s"
        params.append(limit)

        return self.execute_query(query, tuple(params), fetch=True)

    def get_accidents_for_heatmap(self):
        """Obtener coordenadas de accidentes para mapa de calor (últimos 30 días)"""
        query = """
            SELECT latitud, longitud, severidad, ciudad
            FROM accidentes
            WHERE fecha_accidente >= DATE_SUB(NOW(), INTERVAL 30 DAY)
            AND latitud IS NOT NULL 
            AND longitud IS NOT NULL
        """
        return self.execute_query(query, fetch=True)

    # ==================== MÉTODOS DE USUARIOS ====================
    # (sin cambios respecto a tu versión original)
    def get_user_by_nombre(self, nombre):
        """Obtener usuario por nombre (para login)"""
        query = "SELECT * FROM usuarios WHERE nombre = %s AND activo = TRUE"
        result = self.execute_query(query, (nombre,), fetch=True)
        return result[0] if result else None

    def get_user_by_correo(self, correo):
        """Obtener usuario por correo"""
        query = "SELECT * FROM usuarios WHERE correo = %s AND activo = TRUE"
        result = self.execute_query(query, (correo,), fetch=True)
        return result[0] if result else None

    def get_user_by_id(self, user_id):
        """Obtener usuario por ID"""
        query = "SELECT * FROM usuarios WHERE id = %s"
        result = self.execute_query(query, (user_id,), fetch=True)
        return result[0] if result else None

    def create_user(self, nombre, correo, password_hash, role='operador'):
        """Crear nuevo usuario"""
        query = """
            INSERT INTO usuarios (nombre, correo, password, role)
            VALUES (%s, %s, %s, %s)
        """
        return self.execute_query(query, (nombre, correo, password_hash, role))

    def update_last_login(self, user_id):
        """Actualizar último login"""
        query = "UPDATE usuarios SET last_login = NOW() WHERE id = %s"
        self.execute_query(query, (user_id,))

    def increment_failed_attempts(self, user_id):
        """Incrementar intentos fallidos de login"""
        query = """
            UPDATE usuarios 
            SET intentos_fallidos = intentos_fallidos + 1,
                bloqueado_hasta = IF(intentos_fallidos >= 4, DATE_ADD(NOW(), INTERVAL 15 MINUTE), NULL)
            WHERE id = %s
        """
        self.execute_query(query, (user_id,))

    def reset_failed_attempts(self, user_id):
        """Resetear intentos fallidos después de login exitoso"""
        query = "UPDATE usuarios SET intentos_fallidos = 0, bloqueado_hasta = NULL WHERE id = %s"
        self.execute_query(query, (user_id,))

    def get_all_users(self):
        """Obtener todos los usuarios (sin passwords)"""
        query = """
            SELECT id, nombre, correo, role, activo, last_login, creado_en 
            FROM usuarios 
            ORDER BY role, nombre
        """
        return self.execute_query(query, fetch=True)

    # ==================== MÉTODOS DE AUDITORÍA ====================
    def log_action(self, user_id, action, table_affected=None, record_id=None, 
                  detalle=None, old_value=None, new_value=None, ip_address=None, 
                  status='SUCCESS', metodo_http=None):
        """Registrar acción en logs (LEY 1581 - Auditoría)"""
        user_role = None
        if user_id:
            user = self.get_user_by_id(user_id)
            user_role = user['role'] if user else None

        query = """
            INSERT INTO logs (id_usuario, user_role, action, table_affected, 
                             record_id, detalle, old_value, new_value, 
                             ip_address, status, metodo_http)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        self.execute_query(query, (user_id, user_role, action, table_affected, 
                                   record_id, detalle, old_value, new_value, 
                                   ip_address, status, metodo_http))
        
    def get_audit_logs(self, limit=100, user_id=None, action=None, date_from=None, 
                       table_affected=None):
        """Obtener logs de auditoría con filtros"""
        query = """
            SELECT l.*, u.nombre as user_name, u.correo as user_email
            FROM logs l
            LEFT JOIN usuarios u ON l.id_usuario = u.id
            WHERE 1=1
        """
        params = []
        
        if user_id:
            query += " AND l.id_usuario = %s"
            params.append(user_id)
        
        if action:
            query += " AND l.action LIKE %s"
            params.append(f"%{action}%")
        
        if table_affected:
            query += " AND l.table_affected = %s"
            params.append(table_affected)
        
        if date_from:
            query += " AND l.creado_en >= %s"
            params.append(date_from)
        
        query += " ORDER BY l.creado_en DESC LIMIT %s"
        params.append(limit)
        
        return self.execute_query(query, tuple(params), fetch=True)
    
    def get_user_activity(self, user_id, days=30):
        """Obtener actividad de un usuario específico"""
        query = """
            SELECT DATE(creado_en) as fecha, COUNT(*) as acciones
            FROM logs
            WHERE id_usuario = %s 
            AND creado_en >= DATE_SUB(NOW(), INTERVAL %s DAY)
            GROUP BY DATE(creado_en)
            ORDER BY fecha DESC
        """
        return self.execute_query(query, (user_id, days), fetch=True)
    
    # ==================== MÉTODOS DE RETENCIÓN DE DATOS ====================
    
    def get_retention_policies(self):
        """Obtener políticas de retención configuradas"""
        query = "SELECT * FROM data_retention_policy ORDER BY data_type"
        return self.execute_query(query, fetch=True)
    
    def execute_cleanup(self):
        """Ejecutar limpieza de datos antiguos manualmente"""
        query = "CALL cleanup_old_data()"
        try:
            result = self.execute_query(query, fetch=True)
            logger.info("✓ Limpieza de datos ejecutada exitosamente")
            return result
        except Error as e:
            logger.error(f"Error al ejecutar limpieza: {e}")
            raise
    
    def update_retention_policy(self, data_type, retention_days):
        """Actualizar días de retención de una política"""
        query = """
            UPDATE data_retention_policy 
            SET retention_days = %s 
            WHERE data_type = %s
        """
        self.execute_query(query, (retention_days, data_type))
    
    # ==================== MÉTODOS DE ESTADÍSTICAS Y VISTAS ====================
    
    def get_dashboard_stats(self):
        """Obtener estadísticas para el dashboard"""
        query = "SELECT * FROM v_dashboard_stats"
        result = self.execute_query(query, fetch=True)
        return result[0] if result else {}
    
    def get_accidents_by_city(self):
        """Obtener estadísticas de accidentes por ciudad"""
        query = "SELECT * FROM v_accidents_by_city"
        return self.execute_query(query, fetch=True)
    
    def get_accidents_last_30_days(self):
        """Obtener accidentes de los últimos 30 días (vista optimizada)"""
        query = "SELECT * FROM v_accidents_last_30_days ORDER BY fecha_accidente DESC"
        return self.execute_query(query, fetch=True)
    
    def get_accidents_by_month(self, year=None):
        """Obtener accidentes agrupados por mes"""
        if year is None:
            year = datetime.now().year
        
        query = """
            SELECT 
                MONTH(fecha_accidente) as mes,
                MONTHNAME(fecha_accidente) as mes_nombre,
                COUNT(*) as total,
                SUM(CASE WHEN severidad = 'CRÍTICA' THEN 1 ELSE 0 END) as criticos
            FROM accidentes
            WHERE YEAR(fecha_accidente) = %s
            GROUP BY MONTH(fecha_accidente), MONTHNAME(fecha_accidente)
            ORDER BY mes
        """
        return self.execute_query(query, (year,), fetch=True)
    
    def get_accidents_by_hour(self, days=7):
        """Obtener accidentes por hora del día (últimos N días)"""
        query = """
            SELECT 
                HOUR(fecha_accidente) as hora,
                COUNT(*) as total
            FROM accidentes
            WHERE fecha_accidente >= DATE_SUB(NOW(), INTERVAL %s DAY)
            GROUP BY HOUR(fecha_accidente)
            ORDER BY hora
        """
        return self.execute_query(query, (days,), fetch=True)
    
    def get_top_cameras_by_accidents(self, limit=10):
        """Obtener cámaras con más accidentes detectados"""
        query = """
            SELECT 
                c.id,
                c.ip,
                c.ciudad,
                c.direccion,
                COUNT(a.id) as total_accidentes,
                MAX(a.fecha_accidente) as ultimo_accidente
            FROM camaras c
            LEFT JOIN accidentes a ON c.id = a.id_camara
            WHERE a.fecha_accidente >= DATE_SUB(NOW(), INTERVAL 30 DAY)
            GROUP BY c.id, c.ip, c.ciudad, c.direccion
            ORDER BY total_accidentes DESC
            LIMIT %s
        """
        return self.execute_query(query, (limit,), fetch=True)


  # ============================================
    # CONFIGURACIÓN DEL SISTEMA
    # ============================================
    
    def get_config(self, clave):
        """Obtener un valor de configuración"""
        try:
            query = "SELECT valor, tipo FROM sistema_config WHERE clave = %s"
            result = self.execute_query(query, (clave,), fetch=True)
            
            if not result:
                return None
            
            valor = result[0]['valor']
            tipo = result[0]['tipo']
            
            # Convertir al tipo correcto
            if tipo == 'int':
                return int(valor)
            elif tipo == 'float':
                return float(valor)
            elif tipo == 'bool':
                return valor.lower() in ('true', '1', 'yes')
            else:
                return valor
                
        except Exception as e:
            logger.error(f"Error obteniendo config {clave}: {e}")
            return None
    
    def set_config(self, clave, valor):
        """Actualizar un valor de configuración"""
        try:
            query = """
                UPDATE sistema_config 
                SET valor = %s 
                WHERE clave = %s
            """
            self.execute_query(query, (str(valor), clave))
            logger.info(f"✅ Configuración actualizada: {clave} = {valor}")
            return True
        except Exception as e:
            logger.error(f"Error actualizando config {clave}: {e}")
            return False
    
    def get_all_config(self):
        """Obtener toda la configuración"""
        try:
            query = "SELECT clave, valor, tipo, descripcion FROM sistema_config"
            results = self.execute_query(query, fetch=True)
            
            config = {}
            for row in results:
                clave = row['clave']
                valor = row['valor']
                tipo = row['tipo']
                
                # Convertir tipo
                if tipo == 'int':
                    config[clave] = int(valor)
                elif tipo == 'float':
                    config[clave] = float(valor)
                elif tipo == 'bool':
                    config[clave] = valor.lower() in ('true', '1', 'yes')
                else:
                    config[clave] = valor
            
            return config
        except Exception as e:
            logger.error(f"Error obteniendo configuración completa: {e}")
            return {}
        
# Instancia global singleton
db = Database()