import numpy as np
import time
from toolbox.screws_code import joint2twist, forward_kinematics_poe, expScrew
from toolbox.paden_kahan import paden_kahan_one, paden_kahan_three
from toolbox.pardos_gotor import pardos_gotor_one, pardos_gotor_four

# Generación de magnitudes aleatorias
Mag = np.array([np.pi, np.pi/4, np.pi/4, np.pi/4])

# Características mecánicas del robot
po = np.array([0, 0, 0])
pr = np.array([0.4, 0, 0])
pf = np.array([0.65, 0, 0])
pp = np.array([0.65, 0.125, 0])
Point = np.column_stack([po, pr, pf, pp])

AxisY = np.array([0, 1, 0])
Axis = np.column_stack([AxisY, AxisY, AxisY, -AxisY])
JointType = ['rot', 'rot', 'tra', 'rot']

# Inicializar el primer twist
Twist = joint2twist(Axis[:, 0], Point[:, 0], JointType[0])
n = len(JointType)
# Recorrer las demás articulaciones y acumular sus twists
for i in range(1, n):
    twist_i = joint2twist(Axis[:, i], Point[:, i], JointType[i])
    Twist = np.column_stack((Twist, twist_i)) 
# Matriz de transformación Hst0
Hst0 = np.array([[-1, 0, 0, 0.65],[0,0,-1, 0.125],[0,-1, 0, 0],[0,0,0,1]])

# Paso 1: Cinemática directa
TwMag = np.vstack((Twist[:, :4], Mag[:4]))  # 6x4 + 1x4 = 7x4
HstR = forward_kinematics_poe(TwMag)
noap = HstR @ Hst0

# Paso 2-1: PG1 + PG4 + PK1
Theta_STR1 = np.zeros((2, 6))
start_time = time.time()

# Calcular Theta3 con PG1
pkp = noap @ np.linalg.inv(Hst0) @ np.append(pf, 1)
Theta_STR1[0, 2] = pardos_gotor_one(Twist[:, 2:3], pf, pkp[:3])
Theta_STR1[1, 2] = Theta_STR1[0, 2]

# Calcular Theta1 y Theta2 con PG4
pfp = expScrew(np.hstack((Twist[:, 2], Theta_STR1[0, 2]))) @ np.append(pf, 1)
Theta_STR1[0:2, 0:2] = pardos_gotor_four(Twist[:, 0], Twist[:, 1], pfp[:3], pkp[:3])

# Calcular Theta4 con PK1
noapHst0io = noap @ np.linalg.inv(Hst0) @ np.append(po, 1)
for i in range(2):
    pk3p = np.linalg.inv(expScrew(np.hstack((Twist[:, 0], Theta_STR1[i, 0])))) @ noapHst0io
    pk3p = np.linalg.inv(expScrew(np.hstack((Twist[:, 1], Theta_STR1[i, 1])))) @ pk3p
    pk3p = np.linalg.inv(expScrew(np.hstack((Twist[:, 2], Theta_STR1[i, 2])))) @ pk3p
    Theta_STR1[i, 3] = paden_kahan_one(Twist[:, 3:4], po, pk3p[:3])

tIK1 = round((time.time() - start_time) * 1000, 1)
print("Theta_STR1:\n", Theta_STR1)
print(f"Time to solve IK Screw Theory (method 1): {tIK1} ms")

# Paso 2-2: PG1 + PK3 + PK1 + PK1
Theta_STR2 = np.zeros((2, 6))
start_time = time.time()

# Calcular Theta3 con PG1
pkp = noap @ np.linalg.inv(Hst0) @ np.append(pf, 1)
Theta_STR2[0, 2] = pardos_gotor_one(Twist[:, 2:3], pf, pkp[:3])
Theta_STR2[1, :] = Theta_STR2[0, :]

# Calcular Theta2 con PK3
de = np.linalg.norm(pkp[:3] - po)
pfp = expScrew(np.hstack((Twist[:, 2], Theta_STR2[0, 2]))) @ np.append(pf, 1)
Theta_STR2[0:2, 1] = paden_kahan_three(Twist[:, 1:2], pfp[:3], po, de)

# Calcular Theta1 con PK1
for i in range(2):
    pf2p = expScrew(np.hstack((Twist[:, 1], Theta_STR2[i, 1]))) @ pfp
    Theta_STR2[i, 0] = paden_kahan_one(Twist[:, 0:1], pf2p[:3], pkp[:3])

# Calcular Theta4 con PK1
noapHst0io = noap @ np.linalg.inv(Hst0) @ np.append(po, 1)
for i in range(2):
    pk3p = np.linalg.inv(expScrew(np.hstack((Twist[:, 0], Theta_STR2[i, 0])))) @ noapHst0io
    pk3p = np.linalg.inv(expScrew(np.hstack((Twist[:, 1], Theta_STR2[i, 1])))) @ pk3p
    pk3p = np.linalg.inv(expScrew(np.hstack((Twist[:, 2], Theta_STR2[i, 2])))) @ pk3p
    Theta_STR2[i, 3] = paden_kahan_one(Twist[:, 3:4], po, pk3p[:3])

tIK2 = round((time.time() - start_time) * 1000, 1)
print("Theta_STR2:\n", Theta_STR2)
print(f"Time to solve IK Screw Theory (method 2): {tIK2} ms")

# Paso 3: Verificación con FK
for i in range(2):
    TwMagi = np.vstack((Twist[:, :4], Theta_STR1[i, :4]))
    HstRi = forward_kinematics_poe(TwMagi)
    noapi = HstRi @ Hst0
    print(f"FK result STR1, solution {i+1}:\n", noapi)

for i in range(2):
    TwMagi = np.vstack((Twist[:, :4], Theta_STR2[i, :4]))
    HstRi = forward_kinematics_poe(TwMagi)
    noapi = HstRi @ Hst0
    print(f"FK result STR2, solution {i+1}:\n", noapi)
