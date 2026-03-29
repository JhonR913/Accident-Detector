# ============================================================================
# SISTEMA DE ENTRENAMIENTO CNN CON PYTORCH - RTX 5060 Ti
# Dataset: A-Z (26) + 0-9 (10) = 36 clases
# Imágenes: 200x400 px, 100 por carpeta = 3,600 imágenes totales
# PyTorch funciona PERFECTO en Windows con GPU
# ============================================================================

import os
import numpy as np
import cv2
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

print("=" * 80)
print("🚀 SISTEMA DE ENTRENAMIENTO - PYTORCH + GPU")
print("=" * 80)

# ============================================================================
# VERIFICAR GPU
# ============================================================================

def verificar_gpu():
    """Verifica disponibilidad de GPU"""
    print("\n🎮 VERIFICANDO GPU...")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✅ GPU detectada: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"   CUDA version: {torch.version.cuda}")
        return torch.device('cuda')
    else:
        print("⚠️ GPU no detectada, usando CPU")
        print("   Para activar GPU: pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        return torch.device('cpu')

device = verificar_gpu()

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

class Config:
    # Rutas (AJUSTA A TU SISTEMA)
    DATASET_PATH = "C:/Users/Ramirez/Desktop/Nueva carpeta (3)/PLACAS_RECORTADAS"
    MODEL_SAVE_PATH = "C:/Users/Ramirez/Desktop/Nueva carpeta (3)/Modelos"
    
    # Dimensiones optimizadas
    IMG_HEIGHT = 200
    IMG_WIDTH = 400
    
    # Entrenamiento optimizado para GPU
    BATCH_SIZE = 64 if device.type == 'cuda' else 16  # GPU: 64, CPU: 16
    EPOCHS = 150
    LEARNING_RATE = 0.001
    PATIENCE = 20
    
    # Data Augmentation
    ROTATION_RANGE = 3
    ZOOM_RANGE = 0.08
    BRIGHTNESS_RANGE = (0.85, 1.15)
    
    # Split
    TEST_SIZE = 0.15
    RANDOM_STATE = 42
    META_ACCURACY = 0.95
    
    # Device
    DEVICE = device

config = Config()
os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
print(f"\n📁 Dataset: {config.DATASET_PATH}")
print(f"📁 Modelos: {config.MODEL_SAVE_PATH}")
print(f"💻 Device: {config.DEVICE}")
print(f"📦 Batch size: {config.BATCH_SIZE}")
print("=" * 80)

# ============================================================================
# DATASET PERSONALIZADO
# ============================================================================

class PlacasDataset(Dataset):
    """Dataset para placas vehiculares con augmentation"""
    
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convertir a tensor (C, H, W)
        image = torch.from_numpy(image).float()
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ============================================================================
# CARGAR DATOS
# ============================================================================

def cargar_datos():
    """Carga dataset con estructura A-Z, 0-9"""
    print("\n📂 CARGANDO DATASET...")
    
    imagenes = []
    etiquetas = []
    estadisticas = {}
    
    caracteres = sorted([d for d in os.listdir(config.DATASET_PATH)
                        if os.path.isdir(os.path.join(config.DATASET_PATH, d))])
    
    print(f"Clases encontradas: {len(caracteres)}")
    
    for idx, caracter in enumerate(caracteres):
        carpeta = os.path.join(config.DATASET_PATH, caracter)
        archivos = [f for f in os.listdir(carpeta) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        contador = 0
        for archivo in archivos:
            ruta = os.path.join(carpeta, archivo)
            try:
                img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (config.IMG_WIDTH, config.IMG_HEIGHT))
                    # Normalizar
                    img = img.astype(np.float32) / 255.0
                    # Añadir dimensión de canal (1, H, W)
                    img = np.expand_dims(img, axis=0)
                    imagenes.append(img)
                    etiquetas.append(caracter)
                    contador += 1
            except Exception as e:
                print(f"⚠️ Error en {archivo}: {e}")
        
        estadisticas[caracter] = contador
        print(f"  {caracter}: {contador} imágenes")
    
    X = np.array(imagenes)
    y = np.array(etiquetas)
    
    # Codificar etiquetas
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    print("\n" + "=" * 70)
    print(f"✅ Dataset cargado")
    print(f"   Total: {len(X):,} imágenes")
    print(f"   Clases: {len(encoder.classes_)}")
    print(f"   Shape: {X.shape}")
    print("=" * 70)
    
    # Guardar metadata
    metadata = {
        'clases': encoder.classes_.tolist(),
        'num_clases': len(encoder.classes_),
        'img_height': config.IMG_HEIGHT,
        'img_width': config.IMG_WIDTH,
        'estadisticas': estadisticas,
        'fecha': datetime.now().isoformat()
    }
    
    with open(os.path.join(config.MODEL_SAVE_PATH, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    np.save(os.path.join(config.MODEL_SAVE_PATH, 'encoder.npy'), encoder.classes_)
    
    return X, y_encoded, encoder, estadisticas

# ============================================================================
# MODELO CNN
# ============================================================================

class PlacasCNN(nn.Module):
    """Red neuronal convolucional para reconocimiento de caracteres"""
    
    def __init__(self, num_classes):
        super(PlacasCNN, self).__init__()
        
        # Bloque 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(0.25)
        
        # Bloque 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(0.25)
        
        # Bloque 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout2d(0.3)
        
        # Bloque 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.drop4 = nn.Dropout2d(0.3)
        
        # Clasificador
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.drop5 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.drop6 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # Bloque 1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.drop1(x)
        
        # Bloque 2
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.drop2(x)
        
        # Bloque 3
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.drop3(x)
        
        # Bloque 4
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = self.drop4(x)
        
        # Clasificador
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.bn_fc1(self.fc1(x)))
        x = self.drop5(x)
        x = self.relu(self.fc2(x))
        x = self.drop6(x)
        x = self.fc3(x)
        
        return x

# ============================================================================
# ENTRENAMIENTO
# ============================================================================

def entrenar(model, train_loader, val_loader, encoder):
    """Entrena el modelo"""
    
    print("\n🔥 INICIANDO ENTRENAMIENTO...")
    print(f"   Batch size: {config.BATCH_SIZE}")
    print(f"   Épocas: {config.EPOCHS}")
    print(f"   Device: {config.DEVICE}")
    print(f"   Learning rate: {config.LEARNING_RATE}")
    print("=" * 70)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.5, patience=10)
    
    mejor_val_acc = 0.0
    patience_counter = 0
    historial = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(config.EPOCHS):
        # ENTRENAMIENTO
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Época {epoch+1}/{config.EPOCHS}')
        for data, target in pbar:
            data, target = data.to(config.DEVICE), target.to(config.DEVICE)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                            'acc': f'{100.*train_correct/train_total:.2f}%'})
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # VALIDACIÓN
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(config.DEVICE), target.to(config.DEVICE)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Guardar historial
        historial['train_loss'].append(train_loss)
        historial['train_acc'].append(train_acc)
        historial['val_loss'].append(val_loss)
        historial['val_acc'].append(val_acc)
        
        # Actualizar LR
        scheduler.step(val_loss)
        
        print(f'\nÉpoca {epoch+1}/{config.EPOCHS}:')
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # Guardar mejor modelo
        if val_acc > mejor_val_acc:
            mejor_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_PATH, 'mejor_modelo.pth'))
            print(f'  ✅ Mejor modelo guardado (val_acc: {val_acc:.2f}%)')
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f'\n⚠️ Early stopping en época {epoch+1}')
            break
        
        print('-' * 70)
    
    return historial, mejor_val_acc

# ============================================================================
# EVALUACIÓN
# ============================================================================

def evaluar_modelo(model, test_loader, encoder):
    """Evalúa el modelo y genera reporte"""
    print("\n📊 EVALUACIÓN FINAL...")
    
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(config.DEVICE)
            output = model(data)
            _, predicted = output.max(1)
            
            y_true.extend(target.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    # Métricas
    accuracy = 100. * np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)
    
    print("\n" + "=" * 70)
    print("📈 RESULTADOS FINALES")
    print("=" * 70)
    print(f"Precisión: {accuracy:.2f}%")
    
    if accuracy >= config.META_ACCURACY * 100:
        print(f"✅ ¡META ALCANZADA! ({accuracy:.2f}% >= {config.META_ACCURACY*100}%)")
    else:
        print(f"⚠️ Meta no alcanzada ({accuracy:.2f}% < {config.META_ACCURACY*100}%)")
    
    print("\n📋 Reporte de clasificación:")
    print(classification_report(y_true, y_pred, target_names=encoder.classes_, digits=4))
    
    return accuracy, y_true, y_pred

# ============================================================================
# VISUALIZACIÓN
# ============================================================================

def visualizar_resultados(history, accuracy, y_true, y_pred, encoder):
    """Genera visualizaciones"""
    
    fig = plt.figure(figsize=(18, 5))
    
    # Precisión
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(history['train_acc'], label='Train', linewidth=2)
    ax1.plot(history['val_acc'], label='Validation', linewidth=2)
    ax1.axhline(y=config.META_ACCURACY*100, color='r', linestyle='--', label='Meta')
    ax1.set_title(f'Precisión (Final: {accuracy:.2f}%)', fontweight='bold')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Precisión (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Pérdida
    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(history['train_loss'], label='Train', linewidth=2)
    ax2.plot(history['val_loss'], label='Validation', linewidth=2)
    ax2.set_title('Pérdida', fontweight='bold')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Pérdida')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Matriz de confusión (simplificada)
    ax3 = plt.subplot(1, 3, 3)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=False, cmap='Blues', ax=ax3,
                xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    ax3.set_title('Matriz de Confusión', fontweight='bold')
    ax3.set_xlabel('Predicción')
    ax3.set_ylabel('Real')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.MODEL_SAVE_PATH, 'resultados.png'), 
                dpi=300, bbox_inches='tight')
    print(f"\n📊 Gráficos guardados: resultados.png")
    plt.show()

# ============================================================================
# PIPELINE COMPLETO
# ============================================================================

def entrenar_completo():
    """Pipeline completo de entrenamiento"""
    
    # 1. Cargar datos
    X, y, encoder, stats = cargar_datos()
    
    # 2. Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )
    
    print(f"\n📊 Train: {len(X_train):,} | Val: {len(X_val):,}")
    
    # 3. Crear datasets
    train_dataset = PlacasDataset(X_train, y_train)
    val_dataset = PlacasDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, 
                           shuffle=False, num_workers=0)
    
    # 4. Crear modelo
    model = PlacasCNN(num_classes=len(encoder.classes_)).to(config.DEVICE)
    
    print("\n📋 Arquitectura del modelo:")
    print(model)
    print(f"\nParámetros totales: {sum(p.numel() for p in model.parameters()):,}")
    
    # 5. Entrenar
    history, mejor_acc = entrenar(model, train_loader, val_loader, encoder)
    
    # 6. Cargar mejor modelo
    model.load_state_dict(torch.load(os.path.join(config.MODEL_SAVE_PATH, 'mejor_modelo.pth')))
    
    # 7. Evaluar
    accuracy, y_true, y_pred = evaluar_modelo(model, val_loader, encoder)
    
    # 8. Guardar modelo final
    torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_PATH, 'modelo_final.pth'))
    
    # 9. Guardar resultados
    resultados = {
        'fecha': datetime.now().isoformat(),
        'precision_final': float(accuracy),
        'mejor_val_acc': float(mejor_acc),
        'meta_alcanzada': bool(accuracy >= config.META_ACCURACY * 100),
        'epocas': len(history['train_loss']),
        'device': str(config.DEVICE)
    }
    
    with open(os.path.join(config.MODEL_SAVE_PATH, 'resultados.json'), 'w') as f:
        json.dump(resultados, f, indent=2)
    
    # 10. Visualizar
    visualizar_resultados(history, accuracy, y_true, y_pred, encoder)
    
    print("\n" + "=" * 80)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print(f"📊 Precisión: {accuracy:.2f}%")
    print(f"📁 Modelos: {config.MODEL_SAVE_PATH}")
    print("=" * 80)
    
    return model, encoder, accuracy

# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    try:
        print("\n" + "=" * 80)
        print("🚀 INICIANDO ENTRENAMIENTO")
        print("=" * 80)
        
        modelo, encoder, accuracy = entrenar_completo()
        
        print("\n✅ ¡Todo completado exitosamente!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Entrenamiento interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()