import pybullet as p
import pybullet_data
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ik_robot_an5_ur5_unicauca import ik_robot_ur
from toolbox.trajectory import genrate_trajectory_quintic, normalize_angle_rad

def generate_trajectory(p_start, p_end, duration, sampling_time):
    x = genrate_trajectory_quintic(sampling_time, p_start[0], p_end[0], duration)
    y = genrate_trajectory_quintic(sampling_time, p_start[1], p_end[1], duration)
    z = genrate_trajectory_quintic(sampling_time, p_start[2], p_end[2], duration)

    trajectory = []
    for i in range(len(x)):
        homogeneous_matrix = np.eye(4)
        homogeneous_matrix[:3, :3] = np.array([[-1, 0, 0],[0,0,1],[0,1,0]])  # Rotación identidad con ajustes de signo
        homogeneous_matrix[:3, 3] = np.array([x[i], y[i], z[i]])
        trajectory.append(homogeneous_matrix)

    return trajectory, (x, y, z)

def generate_helicoidal_trajectory(p_start, p_end, duration, sampling_time, radius=0.1, turns=3):
    num_samples = int(duration / sampling_time)
    
    # Interpolación en z (de arriba a abajo)
    z = np.linspace(p_start[2], p_end[2], num_samples)
    
    # Ángulo theta desde 0 hasta 2π * número de vueltas
    theta = np.linspace(0, 2 * np.pi * turns, num_samples)
    
    # Coordenadas helicoidales (x constante, y circular)
    y = p_start[1] + radius * np.sin(theta)             # y circular
    x = p_start[0] + radius * np.cos(theta)             # componente circular

    trajectory = []
    for i in range(num_samples):
        homogeneous_matrix = np.eye(4)
        homogeneous_matrix[:3, :3] = np.eye(3)  # Rotación identidad
        homogeneous_matrix[:3, 3] = np.array([x[i], y[i], z[i]])
        trajectory.append(homogeneous_matrix)
    
    return trajectory, (x, y, z)


# Parámetros
duration = 10  # segundos
sampling_time = 1.0 / 240.0  # 240 Hz
p_start = np.array([0.5, 0, 0.5])
p_end   = np.array([0.5, 0, 0.2])

# Generar trayectoria cartesiana y articular
trajectory, xyz_trajectory = generate_helicoidal_trajectory(p_start, p_end, duration, sampling_time)
joint_trajectories = [ik_robot_ur(hm,0) for hm in trajectory]

# --- Gráfico 3D de trayectoria cartesiana ---
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(*xyz_trajectory, label='Trayectoria', color='blue')
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_title('Trayectoria cartesiana 3D')
ax.legend(); plt.show()

# --- Gráfico de señales articulares ---
qs = [[normalize_angle_rad(fila[i]) for fila in joint_trajectories] for i in range(6)]

fig, axs = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
for i in range(6):
    axs[i].plot(qs[i], label=f'q{i+1}')
    axs[i].set_ylabel(f'q{i+1} [rad]')
    axs[i].legend(loc='upper right'); axs[i].grid(True)
axs[-1].set_xlabel('Muestras'); plt.tight_layout(); plt.show()

# --- PyBullet: Simulación ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")

robot_id = p.loadURDF(
    "robots_models/ur5_description/robots/urdf/ur5e.urdf",
    basePosition=[0, 0, 0],
    useFixedBase=True
)

# Detectar articulaciones revolutas
num_joints = p.getNumJoints(robot_id)
controlled_joints = [
    i for i in range(num_joints)
    if p.getJointInfo(robot_id, i)[2] == p.JOINT_REVOLUTE
]

# Crear sliders de cámara y control
slider_cam_dist = p.addUserDebugParameter("Camera Distance", 0.5, 5.0, 2.5)
slider_cam_yaw = p.addUserDebugParameter("Camera Yaw", -180, 180, 50)
slider_cam_pitch = p.addUserDebugParameter("Camera Pitch", -90, 90, -35)
slider_target_x = p.addUserDebugParameter("Target X", -2, 2, 0)
slider_target_y = p.addUserDebugParameter("Target Y", -2, 2, 0)
slider_target_z = p.addUserDebugParameter("Target Z", 0, 2, 0.5)
slider_pause = p.addUserDebugParameter("Simulación (1=Play, 0=Pausa)", 0, 1, 1)
slider_reset = p.addUserDebugParameter("Reiniciar (1=Reset)", 0, 1, 0)

step = 0
reset_flag_last = 0

while p.isConnected():
    sim_flag = p.readUserDebugParameter(slider_pause)
    reset_flag = p.readUserDebugParameter(slider_reset)

    # Actualizar cámara
    cam_params = [p.readUserDebugParameter(s) for s in [
        slider_cam_dist, slider_cam_yaw, slider_cam_pitch,
        slider_target_x, slider_target_y, slider_target_z
    ]]
    p.resetDebugVisualizerCamera(cam_params[0], cam_params[1], cam_params[2], cam_params[3:6])

    # Reinicio
    if reset_flag > 0.5 and reset_flag_last <= 0.5:
        step = 0
        print("▶ Trayectoria reiniciada")
    reset_flag_last = reset_flag

    if sim_flag > 0.5:
        if step < len(joint_trajectories):
            joint_angles = joint_trajectories[step]
            for j, joint_index in enumerate(controlled_joints):
                p.setJointMotorControl2(
                    bodyUniqueId=robot_id,
                    jointIndex=joint_index,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=joint_angles[j],
                    force=500
                )
                
            step += 1
        p.stepSimulation()
    time.sleep(sampling_time)
