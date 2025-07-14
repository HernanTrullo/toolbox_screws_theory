import pybullet as p
import pybullet_data
import time
import math
import numpy as np

# Función para calcular el polinomio de grado 5
def quintic_polynomial(t, p_start, p_end):
    t = np.clip(t, 0, 1)  # Normalizar tiempo entre 0 y 1
    a0 = p_start
    a1 = 0
    a2 = 0
    a3 = 10 * (p_end - p_start)
    a4 = -15 * (p_end - p_start)
    a5 = 6 * (p_end - p_start)
    return a0 + a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5

# Función de cinemática inversa (simulada, reemplaza con tu implementación)
def ik_robot_puma(matrix_homogenea):
    # Asume que matrix_homogenea es una matriz 4x4 (numpy array)
    # Debe devolver un vector de 6 ángulos articulares [theta1, theta2, ..., theta6]
    # Aquí se simula una salida genérica; reemplaza con tu función real
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Placeholder

# Conexión con GUI
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Cargar plano
p.loadURDF("plane.urdf")

# Cargar robot PUMA 560 con base fija
robot_id = p.loadURDF(
    "robots_models/puma560_description/urdf/puma560_robot.urdf",
    basePosition=[0, 0, 0],
    useFixedBase=True
)

# Obtener número de articulaciones
num_joints = p.getNumJoints(robot_id)

# Establecer posición inicial de cámara
p.resetDebugVisualizerCamera(
    cameraDistance=2.5,
    cameraYaw=50,
    cameraPitch=-35,
    cameraTargetPosition=[0, 0, 0.5]
)

# Parámetros de la trayectoria
duration = 10  # segundos
sampling_time = 1.0 / 240.0  # frecuencia de simulación (240 Hz)
start_time = time.time()

# Puntos inicial y final de la trayectoria lineal en 3D
p_start = np.array([0.5, 0.0, 0.5])  # [x_s, y_s, z_s]
p_end = np.array([0.7, 0.2, 0.7])    # [x_e, y_e, z_e]

# Matriz de rotación (orientación constante, identidad por defecto)
rotation_matrix = np.eye(3)  # Matriz identidad 3x3

# Bucle de simulación
while p.isConnected():
    current_time = time.time() - start_time
    if current_time > duration:
        break  # detener después de 10 segundos

    # Calcular tiempo normalizado (0 a 1)
    t_normalized = current_time / duration

    # Calcular posición deseada usando el polinomio de grado 5
    x = quintic_polynomial(t_normalized, p_start[0], p_end[0])
    y = quintic_polynomial(t_normalized, p_start[1], p_end[1])
    z = quintic_polynomial(t_normalized, p_start[2], p_end[2])
    position = np.array([x, y, z])

    # Construir matriz homogénea (4x4)
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = rotation_matrix  # Matriz de rotación
    homogeneous_matrix[:3, 3] = position           # Vector de posición

    # Calcular cinemática inversa
    joint_angles = ik_robot_puma(homogeneous_matrix)

    # Controlar las articulaciones
    for joint_index in range(min(num_joints, len(joint_angles))):
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=joint_angles[joint_index],
            force=500
        )

    p.stepSimulation()
    time.sleep(sampling_time)

# Desconectar
p.disconnect()