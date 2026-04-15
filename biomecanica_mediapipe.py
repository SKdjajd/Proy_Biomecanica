"""
Análisis Biomecánico en Tiempo Real con MediaPipe
--------------------------------------------------
Este script utiliza la cámara de la laptop para detectar puntos del cuerpo humano,
calcular ángulos articulares y analizar el torso en tiempo real.
Utiliza MediaPipe para la detección de poses y OpenCV para el procesamiento de video.
Versión completa con MediaPipe habilitado.
"""

import cv2
import numpy as np
import math
import time
import os

# Importar MediaPipe
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class AnalisisBiomecanico:
    """
    Clase para realizar análisis biomecánico en tiempo real usando la cámara.
    """

    def __init__(self, camera_index=0, width=640, height=480):
        """
        Inicializa el analizador biomecánico.

        Args:
            camera_index (int): Índice de la cámara (0 para cámara predeterminada)
            width (int): Ancho de la resolución de video
            height (int): Alto de la resolución de video
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.lado = 'derecho'  # 'derecho' o 'izquierdo'
        self.angulos = {}

        # Inicializar captura de video
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print("Error: No se puede acceder a la cámara")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Configurar MediaPipe
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def calcular_angulo(self, punto1, punto2, punto3):
        """
        Calcula el ángulo entre tres puntos en grados.

        Args:
            punto1, punto2, punto3: Tuplas (x, y) representando coordenadas

        Returns:
            float: Ángulo en grados
        """
        # Convertir a arrays numpy
        a = np.array(punto1)
        b = np.array(punto2)
        c = np.array(punto3)

        # Vectores
        ba = a - b
        bc = c - b

        # Producto punto y normas
        cos_angulo = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

        # Asegurar que esté en el rango [-1, 1] para evitar errores numéricos
        cos_angulo = np.clip(cos_angulo, -1, 1)

        # Calcular ángulo en grados
        angulo = math.degrees(math.acos(cos_angulo))

        return round(angulo, 1)

    def obtener_puntos_cuerpo(self, landmarks):
        """
        Extrae los puntos clave del cuerpo para el lado especificado.

        Args:
            landmarks: Lista de landmarks de MediaPipe

        Returns:
            dict: Diccionario con coordenadas de puntos clave
        """
        puntos = {}

        if self.lado == 'derecho':
            puntos['hombro'] = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            puntos['codo'] = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            puntos['muneca'] = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            puntos['cadera'] = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            puntos['rodilla'] = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            puntos['tobillo'] = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        else:  # izquierdo
            puntos['hombro'] = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            puntos['codo'] = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            puntos['muneca'] = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            puntos['cadera'] = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            puntos['rodilla'] = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            puntos['tobillo'] = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        return puntos

    def calcular_angulos_articulares(self, puntos):
        """
        Calcula los ángulos articulares principales.

        Args:
            puntos (dict): Puntos clave del cuerpo

        Returns:
            dict: Ángulos calculados
        """
        angulos = {}

        # Ángulo del codo (hombro -> codo -> muñeca)
        angulos['codo'] = self.calcular_angulo(
            puntos['hombro'], puntos['codo'], puntos['muneca']
        )

        # Ángulo de la rodilla (cadera -> rodilla -> tobillo)
        angulos['rodilla'] = self.calcular_angulo(
            puntos['cadera'], puntos['rodilla'], puntos['tobillo']
        )

        # Ángulo del hombro (cadera -> hombro -> codo)
        angulos['hombro'] = self.calcular_angulo(
            puntos['cadera'], puntos['hombro'], puntos['codo']
        )

        # Ángulo de la cadera (hombro -> cadera -> rodilla)
        angulos['cadera'] = self.calcular_angulo(
            puntos['hombro'], puntos['cadera'], puntos['rodilla']
        )

        # Ángulo del tobillo (aproximación: rodilla -> tobillo -> punto proyectado)
        # Para una mejor aproximación necesitaríamos más puntos, pero usamos una proyección
        tobillo_proyectado = [puntos['tobillo'][0], puntos['tobillo'][1] + 0.1]  # Punto abajo
        angulos['tobillo'] = self.calcular_angulo(
            puntos['rodilla'], puntos['tobillo'], tobillo_proyectado
        )

        return angulos

    def analizar_torso(self, puntos):
        """
        Analiza el torso del cuerpo incluyendo ángulo, simetría y postura.

        Args:
            puntos (dict): Puntos clave del cuerpo

        Returns:
            dict: Análisis del torso
        """
        analisis_torso = {}

        # Calcular ángulo del torso (ángulo entre hombros y caderas)
        # Punto medio de hombros
        hombro_izq = np.array([puntos['hombro_izq'][0] if 'hombro_izq' in puntos else puntos.get('hombro', [0.5, 0.25])[0],
                              puntos['hombro_izq'][1] if 'hombro_izq' in puntos else puntos.get('hombro', [0.5, 0.25])[1]])
        hombro_der = np.array([puntos['hombro_der'][0] if 'hombro_der' in puntos else puntos.get('hombro', [0.5, 0.25])[0],
                              puntos['hombro_der'][1] if 'hombro_der' in puntos else puntos.get('hombro', [0.5, 0.25])[1]])
        cadera_izq = np.array([puntos['cadera_izq'][0] if 'cadera_izq' in puntos else puntos.get('cadera', [0.5, 0.5])[0],
                              puntos['cadera_izq'][1] if 'cadera_izq' in puntos else puntos.get('cadera', [0.5, 0.5])[1]])
        cadera_der = np.array([puntos['cadera_der'][0] if 'cadera_der' in puntos else puntos.get('cadera', [0.5, 0.5])[0],
                              puntos['cadera_der'][1] if 'cadera_der' in puntos else puntos.get('cadera', [0.5, 0.5])[1]])

        # Punto medio hombros y caderas
        medio_hombros = (hombro_izq + hombro_der) / 2
        medio_caderas = (cadera_izq + cadera_der) / 2

        # Vector vertical (referencia)
        vector_vertical = np.array([0, -1])  # Hacia arriba

        # Vector del torso (de caderas a hombros)
        vector_torso = medio_hombros - medio_caderas

        # Calcular ángulo del torso con la vertical
        cos_angulo_torso = np.dot(vector_torso, vector_vertical) / (np.linalg.norm(vector_torso) * np.linalg.norm(vector_vertical))
        cos_angulo_torso = np.clip(cos_angulo_torso, -1, 1)
        angulo_torso = math.degrees(math.acos(cos_angulo_torso))

        # Determinar dirección de la inclinación
        if vector_torso[0] > 0.1:  # Inclinación hacia la derecha
            direccion = "derecha"
        elif vector_torso[0] < -0.1:  # Inclinación hacia la izquierda
            direccion = "izquierda"
        else:
            direccion = "vertical"

        analisis_torso['angulo'] = round(angulo_torso, 1)
        analisis_torso['direccion'] = direccion

        # Evaluar postura
        if angulo_torso < 5:
            analisis_torso['postura'] = "Erecta"
            analisis_torso['color_postura'] = (0, 255, 0)  # Verde
        elif angulo_torso < 15:
            analisis_torso['postura'] = "Ligeramente inclinada"
            analisis_torso['color_postura'] = (0, 255, 255)  # Amarillo
        else:
            analisis_torso['postura'] = "Muy inclinada"
            analisis_torso['color_postura'] = (0, 0, 255)  # Rojo

        # Calcular simetría del torso
        # Distancia entre hombro izquierdo y cadera izquierda vs derecha
        dist_izq = np.linalg.norm(hombro_izq - cadera_izq)
        dist_der = np.linalg.norm(hombro_der - cadera_der)
        simetria = abs(dist_izq - dist_der) / max(dist_izq, dist_der) * 100

        analisis_torso['simetria'] = round(simetria, 1)

        if simetria < 5:
            analisis_torso['simetria_texto'] = "Simétrico"
            analisis_torso['color_simetria'] = (0, 255, 0)
        elif simetria < 15:
            analisis_torso['simetria_texto'] = "Ligeramente asimétrico"
            analisis_torso['color_simetria'] = (0, 255, 255)
        else:
            analisis_torso['simetria_texto'] = "Muy asimétrico"
            analisis_torso['color_simetria'] = (0, 0, 255)

        return analisis_torso

    def dibujar_angulos(self, imagen, angulos, analisis_torso=None):
        """
        Dibuja los ángulos calculados y análisis del torso en la imagen.

        Args:
            imagen: Imagen de OpenCV
            angulos (dict): Ángulos a dibujar
            analisis_torso (dict): Análisis del torso (opcional)
        """
        y_pos = 30
        colores = {
            'codo': (0, 255, 255),      # Amarillo cian
            'rodilla': (255, 255, 0),   # Azul cian
            'hombro': (255, 0, 255),    # Magenta
            'cadera': (0, 255, 0),      # Verde
            'tobillo': (255, 165, 0)    # Naranja
        }

        for articulacion, angulo in angulos.items():
            color = colores.get(articulacion, (255, 255, 255))
            texto = f"{articulacion.capitalize()}: {angulo}°"
            cv2.putText(imagen, texto, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_pos += 25

        # Dibujar análisis del torso si está disponible
        if analisis_torso:
            y_pos += 10  # Espacio adicional
            cv2.putText(imagen, "=== ANÁLISIS DEL TORSO ===", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            y_pos += 25

            # Ángulo del torso
            cv2.putText(imagen, f"Ángulo: {analisis_torso['angulo']}° ({analisis_torso['direccion']})",
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_pos += 25

            # Postura
            cv2.putText(imagen, f"Postura: {analisis_torso['postura']}",
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, analisis_torso['color_postura'], 2)
            y_pos += 25

            # Simetría
            cv2.putText(imagen, f"Simetría: {analisis_torso['simetria']}% - {analisis_torso['simetria_texto']}",
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, analisis_torso['color_simetria'], 2)

    def procesar_frame(self, frame):
        """
        Procesa un frame de video para detectar poses y calcular ángulos.

        Args:
            frame: Frame de video de OpenCV

        Returns:
            frame procesado con dibujos
        """
        # Convertir BGR a RGB para MediaPipe
        imagen_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar con MediaPipe
        resultados = self.pose.process(imagen_rgb)

        if resultados.pose_landmarks:
            # Dibujar landmarks
            mp_drawing.draw_landmarks(
                frame, resultados.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
            )

            # Obtener puntos y calcular ángulos
            puntos = self.obtener_puntos_cuerpo(resultados.pose_landmarks.landmark)
            self.angulos = self.calcular_angulos_articulares(puntos)

            # Analizar torso con puntos reales
            puntos_torso = {
                'hombro_izq': [resultados.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                              resultados.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
                'hombro_der': [resultados.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              resultados.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
                'cadera_izq': [resultados.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                              resultados.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y],
                'cadera_der': [resultados.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                              resultados.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            }

            analisis_torso = self.analizar_torso(puntos_torso)

            # Dibujar ángulos y análisis del torso
            self.dibujar_angulos(frame, self.angulos, analisis_torso)
        else:
            cv2.putText(frame, "No se detecta pose", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Mostrar información adicional
        cv2.putText(frame, f"Lado: {self.lado.capitalize()}", (10, frame.shape[0] - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Presiona 'c' para cambiar lado, 'ESC' para salir", (10, frame.shape[0] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return frame

    def ejecutar(self):
        """
        Ejecuta el análisis biomecánico en tiempo real.
        """
        print("Iniciando análisis biomecánico con MediaPipe...")
        print("Controles:")
        print("  - 'c': Cambiar lado (derecho/izquierdo)")
        print("  - 'ESC': Salir")

        if not self.cap.isOpened():
            print("Error: Cámara no disponible")
            return

        cv2.namedWindow("Análisis Biomecánico - MediaPipe", cv2.WINDOW_NORMAL)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error al capturar frame")
                break

            # Procesar frame
            frame_procesado = self.procesar_frame(frame)

            # Mostrar frame
            cv2.imshow("Análisis Biomecánico - MediaPipe", frame_procesado)

            # Manejar controles de teclado
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('c'):
                self.lado = 'izquierdo' if self.lado == 'derecho' else 'derecho'
                print(f"Cambiado a lado {self.lado}")

        # Limpiar recursos
        self.cap.release()
        cv2.destroyAllWindows()
        self.pose.close()

        print("Análisis biomecánico finalizado")

def main():
    """
    Función principal del programa.
    """
    print("=== ANÁLISIS BIOMECÁNICO CON MEDIAPIPE ===")
    print("Este programa detecta puntos del cuerpo y calcula ángulos articulares en tiempo real.")
    print("Utiliza MediaPipe para detección precisa de poses.")

    # Crear instancia del analizador
    analizador = AnalisisBiomecanico()

    # Ejecutar análisis
    try:
        analizador.ejecutar()
    except KeyboardInterrupt:
        print("\nInterrupción detectada, cerrando...")
    except Exception as e:
        print(f"Error durante la ejecución: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
