# 👤 Análisis Biomecánico en Tiempo Real - MediaPipe

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=yellow)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-green?logo=google&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-orange?logo=opencv&logoColor=white)

## 📋 Descripción
**Análisis postural avanzado** con IA que mide **ángulos articulares** y evalúa **postura** en tiempo real usando tu cámara web.

**Para**: Fisioterapia · Ergonomía · Fitness · Rehabilitación

## 🚀 Características Principales

| Métrica | Precisión | Visual |
|---------|-----------|---------|
| **5 Ángulos articulares** | ±2° | Colores |
| **Análisis TORSO** | Ángulo + simetría | Verde/Rojo/Amarillo |
| **Real-time** | 25 FPS | Overlay |
| **Export CSV** | Timestamps | ✅ |

## 📦 Instalación Rápida

```bash
git clone <URL_REPO>
cd Proy_Biomecanica
pip install -r requirements.txt  # ~500MB MediaPipe
```

## 🎮 Uso Inmediato

### ✨ Versión PRO (MediaPipe)
```bash
python biomecanica_mediapipe.py
```

### 💻 Versión Laptop (Fallback)
```bash
python \"python biomecanica_laptop.py\"
```

**Controles teclado:**
```
C = Cambiar lado (🢂 Derecha ↔ Izquierda 🢀)
S = Guardar CSV datos
ESC = Salir
```

## 📊 Métricas Visualizadas
```
┌─ CODO:     145.2° ──┐  ┌─ TORSO:   3.1° (🟢 Erecta)
│ RODILLA:  162.1°    │  │ SIMETRÍA: 2.4% (🟢 Perfecta)
│ HOMBRO:    23.4°    │  └─────────────┘
│ CADERA:   178.9°    │
│ TOBILLO:   89.2°    │
└─────────────────────┘
```

## 🎯 Análisis Torso Inteligente
```
🟢 <5°     = Erecta (Perfecta)
🟡 5-15°   = Ligeramente inclinada
🔴 >15°    = Muy inclinada (CORREGIR)
```

## 💾 Exportación Datos
**CSV generado**: `biomecanica_YYYYMMDD_HHMMSS.csv`
```
timestamp,codo,rodilla,hombro,cadera,tobillo
2024-01-01T14:30:00,145.2,162.1,23.4,178.9,89.2
```

## ⚙️ Requisitos del Sistema
| Requisito | Mínimo |
|-----------|--------|
| **Cámara** | Webcam 720p |
| **RAM** | 1GB libre |
| **GPU** | No requerida |
| **OS** | Win/Linux/Mac |

## 📁 Estructura del Proyecto
```
Proy_Biomecanica/
├── biomecanica_mediapipe.py    # Versión PRO
├── python biomecanica_laptop.py # Fallback
├── requirements.txt
└── README.md
```

## 🔧 Solución de Problemas
```
❌ \"No camera\": Verifica permisos cámara
❌ \"MediaPipe slow\": Baja resolución (320x240)
❌ \"No pose\": Iluminación/ángulo cámara
```

## 📄 Licencia
MIT License - **Libre uso** con atribución.

---

## 👏 Usos Profesionales
- **🏥 Fisioterapia**: Monitoreo rehabilitación
- **💼 Ergonomía**: Análisis oficina
- **🏋️ Fitness**: Corrección técnica ejercicios
- **🔬 Investigación**: Estudios movimiento

⭐ **¡Dale star si mejora tu postura!**
