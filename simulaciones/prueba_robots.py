import pybullet as p
import pybullet_data
import time
import math

# Conexión con GUI
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Cargar plano
p.loadURDF("plane.urdf")

# Cargar robot PUMA 560 con base fija
robot_id = p.loadURDF(
    "robots_models/ur5_description/robots/urdf/ur5e.urdf",
    basePosition=[0, 0, 0],
    useFixedBase=True
)

# Obtener número de articulaciones
num_joints = p.getNumJoints(robot_id)


controlled_joints = [
    i for i in range(num_joints)
    if p.getJointInfo(robot_id, i)[2] == p.JOINT_REVOLUTE
]
# Crear sliders para cada articulación
sliders = []
for i in range(num_joints):
    joint_info = p.getJointInfo(robot_id, i)
    joint_name = joint_info[1].decode("utf-8")
    
    if p.getJointInfo(robot_id, i)[2] == p.JOINT_REVOLUTE:
        # Rango genérico [-pi, pi]; puedes ajustarlo si conoces los límites reales
        slider = p.addUserDebugParameter(paramName=f"{joint_name} (joint {i})",
                                        rangeMin=-math.pi,
                                        rangeMax=math.pi,
                                        startValue=0.0)
        sliders.append(slider)

# Índice del efector final (puedes ajustarlo si conoces el correcto)
end_effector_index = num_joints - 3

# Bucle principal
while p.isConnected():
    # Leer sliders y actualizar articulaciones
    for j, joint_index in enumerate(controlled_joints):
        target_pos = p.readUserDebugParameter(sliders[j])
        p.setJointMotorControl2(robot_id, joint_index, p.POSITION_CONTROL, targetPosition=target_pos)
    
    # Simulación
    p.stepSimulation()
    time.sleep(1. / 240.)

    # Obtener posición cartesiana del efector final
    link_state = p.getLinkState(robot_id, end_effector_index, computeForwardKinematics=True)
    pos = link_state[0]
    print(f"Posición del efector final: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}", end='\r')
