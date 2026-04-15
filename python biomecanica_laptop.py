"""
Sistema de análisis biomecánico en laptop
-----------------------------------------
Captura video desde cámara local, detecta el esqueleto humano,
calcula ángulos articulares y muestra el movimiento con animación.
Mejorado con más ángulos, logging de datos y controles.
Nota: MediaPipe no está disponible en este entorno, se simula la funcionalidad.
"""

import cv2
import numpy as np
import csv
import os
import sys
from datetime import datetime

# Configurar OpenCV para usar backend GTK en lugar de Qt
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['DISPLAY'] = ':0'
cv2.setUseOptimized(True)
cv2.startWindowThread()

# Simulación de MediaPipe ya que no se puede instalar
class FakeMediaPipe:
    class solutions:
        class drawing_utils:
            @staticmethod
            def draw_landmarks(image, landmarks, connections, spec1, spec2):
                # Simular dibujo de landmarks
                pass

            class DrawingSpec:
                def __init__(self, color=(0, 255, 0), thickness=2, circle_radius=3):
                    pass

        class pose:
            class PoseLandmark:
                RIGHT_SHOULDER = 12
                RIGHT_ELBOW = 14
                RIGHT_WRIST = 16
                RIGHT_HIP = 24
                RIGHT_KNEE = 26
                RIGHT_ANKLE = 28
                LEFT_SHOULDER = 11
                LEFT_ELBOW = 13
                LEFT_WRIST = 15
                LEFT_HIP = 23
                LEFT_KNEE = 25
                LEFT_ANKLE = 27

            POSE_CONNECTIONS = []

            class Pose:
                def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
                    pass

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc_val, exc_tb):
                    pass

                def process(self, image):
                    # Simular resultados de pose
                    class FakeLandmark:
                        def __init__(self, x, y):
                            self.x = x
                            self.y = y

                    class FakePoseLandmarks:
                        def __init__(self):
                            # Simular landmarks con posiciones aleatorias
                            self.landmark = [
                                FakeLandmark(0.5, 0.5) for _ in range(33)  # 33 landmarks típicos
                            ]

                    class FakeResults:
                        def __init__(self):
                            self.pose_landmarks = FakePoseLandmarks()

                    return FakeResults()

mp_drawing = FakeMediaPipe.solutions.drawing_utils
mp_pose = FakeMediaPipe.solutions.pose

class PoseAnalyzer:
    """
    Clase para analizar poses biomecánicas en tiempo real.
    """

    def __init__(self, camera_index=0, width=640, height=480):
        """
        Inicializa el analizador de poses.

        Args:
            camera_index (int): Índice de la cámara.
            width (int): Ancho de la resolución.
            height (int): Alto de la resolución.
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.side = 'RIGHT'  # 'RIGHT' o 'LEFT'
        self.data_log = []  # Lista para almacenar datos
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print("Cámara no disponible, usando imagen simulada.")
            self.use_simulated = True
        else:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.use_simulated = False

    def calcular_angulo(self, p1, p2, p3):
        """
        Calcula el ángulo entre tres puntos usando producto punto.

        Args:
            p1, p2, p3: Tuplas (x, y)

        Returns:
            int: Ángulo en grados.
        """
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)

        ab = a - b
        cb = c - b
        dot = np.dot(ab, cb)
        norma = np.linalg.norm(ab) * np.linalg.norm(cb)
        if norma == 0:
            return 0
        angulo = np.degrees(np.arccos(np.clip(dot / norma, -1, 1)))
        return int(angulo)

    def get_landmarks(self, landmarks, side):
        """
        Obtiene coordenadas de landmarks para el lado especificado.

        Args:
            landmarks: Lista de landmarks de MediaPipe.
            side (str): 'RIGHT' o 'LEFT'.

        Returns:
            dict: Diccionario con coordenadas de puntos clave.
        """
        prefix = 'RIGHT' if side == 'RIGHT' else 'LEFT'
        return {
            'shoulder': [landmarks[getattr(mp_pose.PoseLandmark, f'{prefix}_SHOULDER')].x,
                         landmarks[getattr(mp_pose.PoseLandmark, f'{prefix}_SHOULDER')].y],
            'elbow': [landmarks[getattr(mp_pose.PoseLandmark, f'{prefix}_ELBOW')].x,
                      landmarks[getattr(mp_pose.PoseLandmark, f'{prefix}_ELBOW')].y],
            'wrist': [landmarks[getattr(mp_pose.PoseLandmark, f'{prefix}_WRIST')].x,
                      landmarks[getattr(mp_pose.PoseLandmark, f'{prefix}_WRIST')].y],
            'hip': [landmarks[getattr(mp_pose.PoseLandmark, f'{prefix}_HIP')].x,
                    landmarks[getattr(mp_pose.PoseLandmark, f'{prefix}_HIP')].y],
            'knee': [landmarks[getattr(mp_pose.PoseLandmark, f'{prefix}_KNEE')].x,
                     landmarks[getattr(mp_pose.PoseLandmark, f'{prefix}_KNEE')].y],
            'ankle': [landmarks[getattr(mp_pose.PoseLandmark, f'{prefix}_ANKLE')].x,
                      landmarks[getattr(mp_pose.PoseLandmark, f'{prefix}_ANKLE')].y],
        }

    def calculate_angles(self, points):
        """
        Calcula ángulos articulares.

        Args:
            points (dict): Puntos clave.

        Returns:
            dict: Ángulos calculados.
        """
        angles = {}
        # Ángulos existentes
        angles['elbow'] = self.calcular_angulo(points['shoulder'], points['elbow'], points['wrist'])
        angles['knee'] = self.calcular_angulo(points['hip'], points['knee'], points['ankle'])

        # Nuevos ángulos
        # Para hombro: usamos cadera, hombro, codo (aproximación para flexión/extensión)
        angles['shoulder'] = self.calcular_angulo(points['hip'], points['shoulder'], points['elbow'])
        # Para cadera: usamos hombro, cadera, rodilla
        angles['hip'] = self.calcular_angulo(points['shoulder'], points['hip'], points['knee'])
        # Para tobillo: usamos rodilla, tobillo, y un punto imaginario (aproximación)
        # Para dorsiflexión/plantarflexión, necesitamos más puntos, pero aproximamos con rodilla, tobillo, y un punto extendido
        # Aquí usamos rodilla, tobillo, y proyectamos un punto
        ankle_extended = [points['ankle'][0], points['ankle'][1] - 0.1]  # Punto imaginario abajo
        angles['ankle'] = self.calcular_angulo(points['knee'], points['ankle'], ankle_extended)

        return angles

    def draw_angles(self, image, angles):
        """
        Dibuja los ángulos en la imagen.

        Args:
            image: Imagen de OpenCV.
            angles (dict): Ángulos a dibujar.
        """
        y_offset = 50
        colors = {
            'elbow': (0, 255, 255),
            'knee': (255, 255, 0),
            'shoulder': (255, 0, 255),
            'hip': (0, 255, 0),
            'ankle': (255, 165, 0)
        }
        for joint, angle in angles.items():
            color = colors.get(joint, (255, 255, 255))
            cv2.putText(image, f"{joint.capitalize()}: {angle}°", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 30

    def save_data(self):
        """
        Guarda los datos acumulados en un archivo CSV.
        """
        if not self.data_log:
            print("No hay datos para guardar.")
            return
        filename = f"biomecanica_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['timestamp'] + list(self.data_log[0]['angles'].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for entry in self.data_log:
                row = {'timestamp': entry['timestamp']}
                row.update(entry['angles'])
                writer.writerow(row)
        print(f"Datos guardados en {filename}")
        self.data_log = []  # Limpiar log después de guardar

    def run(self):
        """
        Ejecuta el análisis en tiempo real.
        """
        cv2.destroyAllWindows()  # Cerrar cualquier ventana previa
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            frame_count = 0
            while True:
                if self.use_simulated:
                    # Crear imagen simulada
                    frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    cv2.putText(frame, "Modo Simulado", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    ret = True
                else:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("No se detecta cámara.")
                        break

                # Convertimos BGR a RGB para MediaPipe
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                results = pose.process(image)

                # De vuelta a BGR para OpenCV
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                angles = {}
                try:
                    if results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark
                        points = self.get_landmarks(landmarks, self.side)
                        angles = self.calculate_angles(points)

                        # Log data cada 10 frames
                        if frame_count % 10 == 0:
                            self.data_log.append({
                                'timestamp': datetime.now().isoformat(),
                                'angles': angles
                            })

                        # Dibuja el esqueleto
                        mp_drawing.draw_landmarks(
                            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                        )

                except Exception as e:
                    print(f"Error procesando frame: {e}")

                # Dibuja ángulos
                self.draw_angles(image, angles)

                # Mostrar lado actual y modo
                cv2.putText(image, f"Lado: {self.side}", (20, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(image, f"Modo: {'Simulado' if self.use_simulated else 'Real'}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Mostramos la ventana (ya creada, solo actualizar)
                cv2.imshow("Análisis biomecánico - Laptop", image)

                # Controles de teclado
                key = cv2.waitKey(5) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord('s'):  # Guardar datos
                    self.save_data()
                elif key == ord('l'):  # Cambiar lado
                    self.side = 'LEFT' if self.side == 'RIGHT' else 'RIGHT'
                    print(f"Cambiado a lado {self.side}")

                frame_count += 1

        if not self.use_simulated:
            self.cap.release()
        cv2.destroyAllWindows()

# ---- Ejecución principal ----
if __name__ == "__main__":
    # Verificar si ya hay una instancia corriendo usando un archivo de bloqueo
    lock_file = "biomecanica_lock.tmp"
    if os.path.exists(lock_file):
        print("Ya hay una instancia del script corriendo. Saliendo...")
        sys.exit(1)
    else:
        with open(lock_file, 'w') as f:
            f.write(str(os.getpid()))

    try:
        cv2.destroyAllWindows()  # Cerrar cualquier ventana previa antes de iniciar
        cv2.namedWindow("Análisis biomecánico - Laptop", cv2.WINDOW_NORMAL)  # Crear ventana una vez
        analyzer = PoseAnalyzer()
        analyzer.run()
    except KeyboardInterrupt:
        print("Interrupción detectada, cerrando ventanas y liberando recursos...")
        cv2.destroyAllWindows()
        if hasattr(analyzer, 'cap') and analyzer.cap.isOpened():
            analyzer.cap.release()
    finally:
        # Limpiar archivo de bloqueo
        if os.path.exists(lock_file):
            os.remove(lock_file)
