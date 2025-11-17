import hashlib
from database import db

def create_admin():
    """Crear usuario administrador por defecto"""
    
    # Datos del admin
    nombre = "Administrador"
    correo = "admin@sistema.local"
    password = "Admin123!"  # ⚠️ CAMBIAR DESPUÉS DEL PRIMER LOGIN
    rol = "admin"
    
    # Hash de la contraseña
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    try:
        # Verificar si ya existe
        existing = db.get_user_by_correo(correo)
        
        if existing:
            print(f"❌ El usuario {correo} ya existe")
            print(f"   ID: {existing['id']}")
            print(f"   Nombre: {existing['nombre']}")
            print(f"   Rol: {existing['role']}")
            return
        
        # Crear usuario
        user_id = db.create_user(
            nombre=nombre,
            correo=correo,
            password_hash=password_hash,
            role=rol
        )
        
        print("=" * 60)
        print("✅ USUARIO ADMINISTRADOR CREADO EXITOSAMENTE")
        print("=" * 60)
        print(f"📧 Correo: {correo}")
        print(f"🔑 Contraseña: {password}")
        print(f"👤 Nombre: {nombre}")
        print(f"🎖️  Rol: {rol}")
        print(f"🆔 ID: {user_id}")
        print("=" * 60)
        print("⚠️  IMPORTANTE: Cambia la contraseña después del primer login")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error creando usuario: {e}")

def create_test_users():
    """Crear usuarios de prueba para todos los roles"""
    
    users = [
        {
            'nombre': 'Operador Test',
            'correo': 'operador@test.com',
            'password': 'Operador123!',
            'rol': 'operador'
        },
        {
            'nombre': 'Emergencias Test',
            'correo': 'emergencias@test.com',
            'password': 'Emergencias123!',
            'rol': 'emergencias'
        }
    ]
    
    print("\n📋 CREANDO USUARIOS DE PRUEBA...\n")
    
    for user_data in users:
        try:
            # Verificar si existe
            existing = db.get_user_by_correo(user_data['correo'])
            if existing:
                print(f"⏭️  Usuario {user_data['correo']} ya existe")
                continue
            
            # Hash de contraseña
            password_hash = hashlib.sha256(user_data['password'].encode()).hexdigest()
            
            # Crear
            user_id = db.create_user(
                nombre=user_data['nombre'],
                correo=user_data['correo'],
                password_hash=password_hash,
                role=user_data['rol']
            )
            
            print(f"✅ {user_data['rol'].upper()}")
            print(f"   📧 Correo: {user_data['correo']}")
            print(f"   🔑 Contraseña: {user_data['password']}")
            print(f"   🆔 ID: {user_id}\n")
            
        except Exception as e:
            print(f"❌ Error creando {user_data['nombre']}: {e}\n")

if __name__ == '__main__':
    print("\n🚀 INICIALIZANDO USUARIOS DEL SISTEMA\n")
    
    # Crear admin
    create_admin()
    
    # Crear usuarios de prueba
    create_test_users()
    
    print("\n✅ PROCESO COMPLETADO\n")