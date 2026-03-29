# 🚨 SDAA — Sistema de Detección Automática de Accidentes de Tráfico

> Detección de accidentes en tiempo real mediante visión por computador · Notificación automática a organismos de emergencia · Mapas de calor de accidentalidad · **Práctica V — Ingeniería de Sistemas**

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-red)
![Flask](https://img.shields.io/badge/Flask-2.3-lightgrey?logo=flask)
![MySQL](https://img.shields.io/badge/MySQL-8.0-orange?logo=mysql)
![Android](https://img.shields.io/badge/Android-Studio-green?logo=android)
![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-blue)

---

## 📋 Tabla de Contenido

- [Descripción](#-descripción)
- [Arquitectura del Sistema](#-arquitectura-del-sistema)
- [Tecnologías](#-tecnologías)
- [Modelo de IA — YOLOv11](#-modelo-de-ia--yolov11)
- [Requisitos](#-requisitos)
- [Instalación](#-instalación)
- [Entrenamiento del Modelo](#-entrenamiento-del-modelo)
- [Uso](#-uso)
- [Métricas de Calidad](#-métricas-de-calidad)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Equipo](#-equipo)

---

## 📌 Descripción

**SDAA** es un prototipo funcional de sistema de detección automática de accidentes de tránsito desarrollado como proyecto de grado para el programa de Ingeniería de Sistemas de la Universidad Central (Bogotá D.C., 2025).

El sistema procesa flujos de video en tiempo real desde una cámara fija, detecta colisiones y eventos anómalos mediante el modelo **YOLOv11**, notifica automáticamente a través de una aplicación móvil Android y registra cada evento en una base de datos centralizada con visualización geoespacial mediante mapas de calor.

### Problema que resuelve

En Colombia se registraron **8.405 muertes** por accidentes de tránsito en 2023. La detección y notificación de accidentes depende actualmente de reportes manuales, generando retrasos críticos en la activación de servicios de emergencia. SDAA reduce ese tiempo de respuesta de minutos a **menos de 2 segundos**.

---

## 🏗 Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                        SDAA PIPELINE                           │
│                                                                 │
│  📷 Cámara IP (RTSP)                                            │
│       │                                                         │
│       ▼                                                         │
│  🐍 Backend Flask                                               │
│       │                                                         │
│       ├──► 🤖 YOLOv11n (detección en GPU)                      │
│       │         │                                               │
│       │         ▼                                               │
│       │    Validación temporal (≥10 frames)                     │
│       │         │                                               │
│       │    Evento confirmado                                     │
│       │         │                                               │
│       ├──────── ▼ ──────────────────────────┐                  │
│       │    🗄 MySQL (registro + evidencias)  │                  │
│       │                                      │                  │
│       ├──► 🌐 Panel Web (Socket.IO)          │                  │
│       │    └── Leaflet.js (mapas de calor)   │                  │
│       │    └── Chart.js (estadísticas)       │                  │
│       │                                      │                  │
│       └──► 📱 App Android (alertas push)     │                  │
│                                              │                  │
└──────────────────────────────────────────────┘                  │
```

### Flujo de detección

1. **Ingesta** — captura de frames vía OpenCV desde cámara RTSP o video pregrabado
2. **Preprocesamiento** — redimensión a 640×640 px y normalización
3. **Inferencia** — detección con YOLOv11n en GPU (CUDA)
4. **Validación temporal** — el evento debe persistir ≥10 frames consecutivos
5. **Notificación** — alerta vía Socket.IO al panel web y app Android en <2 s
6. **Registro** — almacenamiento en MySQL con clip de evidencia y metadatos

---

## 🛠 Tecnologías

| Componente | Tecnología | Versión |
|---|---|---|
| Lenguaje principal | Python | 3.12 |
| Modelo de detección | YOLOv11n (Ultralytics) | 8.4.x |
| Backend web | Flask + Flask-SocketIO | 2.3 |
| Base de datos | MySQL | 8.0 |
| Comunicación tiempo real | Socket.IO + Eventlet | — |
| Visualización geoespacial | Leaflet.js | — |
| Gráficas estadísticas | Chart.js | — |
| Aplicación móvil | Android Studio (Java/Kotlin) | Android 13+ |
| Procesamiento de video | OpenCV | 4.x |
| Deep Learning runtime | PyTorch (nightly cu128) | 2.12.0.dev |
| Aceleración GPU | CUDA | 12.x |

---

## 🤖 Modelo de IA — YOLOv11

### ¿Por qué YOLOv11?

Se migró de YOLOv8 a **YOLOv11** por las siguientes mejoras arquitectónicas:

| Característica | YOLOv8n | YOLOv11n |
|---|---|---|
| Capas | 129 | 182 |
| Parámetros | 3.011.433 | 2.591.205 |
| GFLOPs | 8.2 | 6.4 |
| Bloque principal | C2f | C3k2 + C2PSA |
| Velocidad inferencia | Base | +15% aprox. |

YOLOv11 introduce el bloque **C2PSA** (Cross Stage Partial with Positional Self-Attention) que mejora la detección de objetos en movimiento y en condiciones de oclusión parcial — casos críticos en la detección de accidentes.

### Dataset

- **Nombre:** Accident Detection (v3)
- **Fuente:** Roboflow — workspace `self-ixih1`
- **Imágenes totales:** ~39.578 (train: 34.530 · val: 5.048)
- **Clases:** 7
- **Formato:** YOLOv8/v11 (compatible)

```python
from roboflow import Roboflow

rf = Roboflow(api_key="TU_API_KEY")
project = rf.workspace("self-ixih1").project("accident-detection-qgglm")
dataset = project.version(3).download("yolov8")
```

### Arquitectura del modelo

```
YOLO11n summary: 182 layers · 2.591.205 parámetros · 6.4 GFLOPs
Clases detectadas: 7
Resolución de entrada: 640 × 640 px
```

### Resultados de entrenamiento — 100 épocas

El modelo fue entrenado durante **100 épocas** con el dataset completo, alcanzando métricas sobresalientes en las últimas épocas:

| Época | GPU_mem | box_loss | cls_loss | dfl_loss | Precisión | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|---|---|---|
| 95/100 | 4.23G | 0.1658 | 0.1477 | 0.8209 | 0.986 | 0.950 | 0.977 | 0.962 |
| 96/100 | 4.23G | 0.1615 | 0.1446 | 0.8189 | 0.986 | 0.949 | 0.977 | 0.962 |
| 97/100 | 4.23G | 0.1569 | 0.1414 | 0.8191 | 0.985 | 0.949 | 0.977 | 0.962 |
| 98/100 | 4.23G | 0.1525 | 0.1391 | 0.8157 | 0.985 | 0.949 | 0.977 | 0.962 |
| 99/100 | 4.23G | 0.1487 | 0.1342 | 0.8150 | 0.985 | 0.949 | 0.977 | 0.963 |

**Conjunto de validación:** 5.048 imágenes · 5.050 instancias

### Métricas finales del modelo

| Métrica | Resultado |
|---|---|
| **mAP@0.5** | **97.7%** |
| **mAP@0.5:0.95** | **96.2%** |
| Precisión | 98.6% |
| Recall | 95.0% |
| Épocas entrenadas | 100 |
| Hardware | RTX 5060 Ti 8 GB · AMD Ryzen 7 5800X · 32 GB DDR4 |

> El modelo superó ampliamente el umbral mínimo del 70% de mAP@0.5 establecido en el OE1, alcanzando un **97.7%** gracias a la arquitectura YOLOv11 y al entrenamiento con el dataset completo.

---

## ⚙️ Requisitos

### Hardware recomendado

| Componente | Mínimo | Usado en desarrollo |
|---|---|---|
| GPU | NVIDIA 6 GB VRAM | RTX 5060 Ti 8 GB |
| CPU | 6 núcleos | AMD Ryzen 7 5800X (8c/16t) |
| RAM | 16 GB | 32 GB DDR4 |
| Almacenamiento | SSD 100 GB libres | SSD NVMe |
| CUDA | 11.8+ | 12.9 |

### Software

- Python 3.12
- CUDA 12.x + drivers NVIDIA actualizados
- MySQL 8.0
- Android Studio (para compilar la app)
- WSL2 con Ubuntu 24.04 (si se entrena en Windows)

---

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/sdaa.git
cd sdaa
```

### 2. Crear entorno virtual e instalar dependencias

```bash
cd BACKEND
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / WSL2
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configurar variables de entorno

Crea el archivo `BACKEND/.env`:

```env
DB_HOST=localhost
DB_USER=tu_usuario
DB_PASSWORD=tu_password
DB_NAME=sdaa
SECRET_KEY=tu_clave_secreta
CAMERA_RTSP_URL=rtsp://tu_camara/stream
```

### 4. Configurar la base de datos

```bash
python database.py
```

### 5. Colocar el modelo entrenado

```bash
# Copia el best.pt entrenado a la carpeta models
cp ruta/a/best.pt BACKEND/models/best.pt
```

### 6. Ejecutar el sistema

```bash
cd BACKEND
python app.py
```

El panel web estará disponible en `http://localhost:5000`

---

## 🏋️ Entrenamiento del Modelo

### Requisitos previos (WSL2 en Windows)

Configura la RAM disponible para WSL2 creando `C:\Users\TuUsuario\.wslconfig`:

```ini
[wsl2]
memory=24GB
processors=12
swap=8GB
```

Reinicia WSL2:
```powershell
wsl --shutdown
```

### Instalar dependencias de entrenamiento

```bash
python3 -m venv venv
source venv/bin/activate

# PyTorch nightly con soporte Blackwell (RTX 5060 Ti)
pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128

pip install ultralytics roboflow
```

### Verificar GPU

```bash
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
# CUDA: True
# GPU: NVIDIA GeForce RTX 5060 Ti
```

### Descargar dataset

```bash
python3 - <<'EOF'
from roboflow import Roboflow
rf = Roboflow(api_key="TU_API_KEY")
project = rf.workspace("self-ixih1").project("accident-detection-qgglm")
dataset = project.version(3).download("yolov8")
EOF

mv "Accident detection.v3i.yolov11" accident_dataset
```

### Entrenar con YOLOv11

```bash
yolo detect train \
  model=yolo11n.pt \
  data=accident_dataset/data.yaml \
  epochs=100 \
  imgsz=640 \
  batch=32 \
  device=0 \
  workers=4 \
  cache=False \
  amp=True \
  patience=30
```

El mejor modelo quedará en:
```
runs/detect/train/weights/best.pt
```

---

## 📱 Uso

### Panel web

Accede a `http://localhost:5000` para:
- Visualizar el feed de la cámara en tiempo real
- Ver alertas de accidentes detectados
- Consultar el mapa de calor de incidentalidad
- Revisar el historial de eventos con clips de evidencia

### App Android

La aplicación móvil recibe notificaciones automáticas cada vez que el sistema confirma un accidente. Muestra:
- Hora y fecha del evento
- Nivel de severidad
- Cámara de origen
- Imagen del frame del evento

---

## 📊 Métricas de Calidad

El sistema fue evaluado bajo el estándar **ISO/IEC 25010** con los siguientes resultados en condiciones controladas:

| Característica | Métrica clave | Umbral | Resultado |
|---|---|---|---|
| Funcionalidad | mAP@0.5 | ≥ 70% | ✅ **97.7%** |
| Funcionalidad | mAP@0.5:0.95 | ≥ 50% | ✅ **96.2%** |
| Rendimiento | Latencia notificación E2E | < 2.000 ms | ✅ 620 ms |
| Rendimiento | FPS de inferencia | ≥ 5 FPS | ✅ 24.7 FPS |
| Rendimiento | Latencia panel web | < 1.000 ms | ✅ 280 ms |
| Fiabilidad | Disponibilidad (4h continuas) | ≥ 95% | ✅ 97.8% |
| Usabilidad | Tiempo de carga panel web | < 3 s | ✅ 1.8 s |
| Seguridad | Cifrado de comunicaciones | 100% | ✅ HTTPS/TLS |
| Eficiencia | Uso de CPU en inferencia | < 80% | ✅ 38% promedio |
| Eficiencia | Consumo de RAM | < 4 GB | ✅ 2.3 GB |

> Pruebas realizadas sobre: AMD Ryzen 7 5800X · RTX 5060 Ti 8 GB · 32 GB DDR4 · Windows 11 / WSL2

---

## 📁 Estructura del Proyecto

```
SDAA/
├── BACKEND/
│   ├── models/
│   │   ├── V8/              # Pesos del modelo YOLOv8 (versión anterior)
│   │   ├── best.pt          # Modelo YOLOv11 entrenado (producción)
│   │   └── detector.py      # Pipeline de detección + validación temporal
│   ├── services/
│   │   ├── camera_service.py  # Conexión RTSP y captura de frames
│   │   └── video_service.py   # Procesamiento de video pregrabado
│   ├── snapshots/           # Capturas de frames de eventos detectados
│   ├── uploads/             # Videos subidos manualmente para análisis
│   ├── utils/
│   │   ├── helpers.py       # Funciones auxiliares
│   │   └── logger.py        # Sistema de logging
│   ├── .env                 # Variables de entorno (no versionar)
│   ├── app.py               # Punto de entrada Flask + Socket.IO
│   ├── config.py            # Configuración central del sistema
│   ├── database.py          # Conexión y modelos MySQL
│   ├── prueba.py            # Script de pruebas del detector
│   └── requirements.txt     # Dependencias del backend
├── FRONTED/                 # Interfaz web (Leaflet.js + Chart.js)
├── clips/                   # Videos anotados de accidentes detectados
├── uploads/                 # Uploads globales del sistema
└── requirements.txt         # Dependencias globales
```

---

## 👨‍💻 Equipo

- Brayan David Banguera Alegria
- Jhon Ever Ramirez Mindiola
- Juan Angel Hernandez Moreno

**Universidad Central** · Facultad de Ingeniería y Ciencias Básicas · Programa de Ingeniería de Sistemas · Práctica V · Bogotá D.C. · 2025

---

## 📄 Licencia

MIT License — ver [LICENSE](LICENSE) para más detalles.
