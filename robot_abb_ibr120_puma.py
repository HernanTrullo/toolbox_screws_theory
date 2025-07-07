# Copyright (C) 2025 Hernan Trullo
#
# This file is part of [Nombre de tu proyecto o biblioteca].
#
# [Nombre de tu proyecto o biblioteca] is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# [Nombre de tu proyecto o biblioteca] is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from time import time
from toolbox.screws_code import forward_kinematics_poe, joint2twist, expScrew
from toolbox.paden_kahan import paden_kahan_three, paden_kahan_two, paden_kahan_one

# Número de grados de libertad
n = 6
# Generar magnitudes aleatorias theta1-theta6
Mag = np.ones(6)*np.pi

# Características mecánicas del robot IRB120
po = np.array([0, 0, 0])
pk = np.array([0, 0, 0.290])
pr = np.array([0, 0, 0.560])
pf = np.array([0.302, 0, 0.630])
pp = np.array([0.302, 0, 0.470])

AxisX = np.array([1, 0, 0])
AxisY = np.array([0, 1, 0])
AxisZ = np.array([0, 0, 1])

# Puntos en las articulaciones
Point = np.column_stack((po, pk, pr, pf, pf, pf))

# Tipos de articulaciones
JointType = ['rot', 'rot', 'rot', 'rot', 'rot', 'rot']

# Ejes de las articulaciones
Axis = np.column_stack((AxisZ, AxisY, AxisY, AxisX, AxisY, -AxisZ))

# Inicializar el primer twist
Twist = joint2twist(Axis[:, 0], Point[:, 0], JointType[0])
# Recorrer las demás articulaciones y acumular sus twists
for i in range(1, n):
    twist_i = joint2twist(Axis[:, i], Point[:, i], JointType[i])
    Twist = np.column_stack((Twist, twist_i)) 
    

Hst0 = np.array([[-1, 0, 0, 0.3020],[0,1,0,0],[0,0,-1, 0.47],[0,0,0,1]])
#Aplicar Forward Kinematics POE
TwMag = np.vstack((Twist, Mag))  # Apila Twist (6xn) y Mag (1xn)
HstR = forward_kinematics_poe(TwMag)
noap = HstR @ Hst0

# matriz homogénea del efector final
#print("HstR (matriz homogénea final del efector):\n", noap)

# Resolución de la cinemática inversa por PK3 + PK2 + PK2 + PK1
Theta_STR1 = np.zeros((8, n))
start_time = time()

# Paso 1: Theta3 por PK3
noapHst0if = noap @ np.linalg.inv(Hst0) @ np.append(pf, 1)
pkp = noapHst0if[:3]
de = np.linalg.norm(pkp - pk)
t3 = paden_kahan_three(Twist[:, 2:3], pf, pk, de)
Theta_STR1[0:4, 2] = t3[0]
Theta_STR1[4:8, 2] = t3[1]

# Paso 2: Theta1 y Theta2 por PK2
for i in [0, 4]:
    pfpt = expScrew(np.append(Twist[:, 2].reshape(-1, 1), Theta_STR1[i, 2])) @ np.append(pf, 1)
    pfp = pfpt[:3]
    t1t2 = paden_kahan_two(Twist[:, 0], Twist[:, 1], pfp, pkp)
    Theta_STR1[i, 0:2]     = t1t2[0]
    Theta_STR1[i + 1, 0:2] = t1t2[0]
    Theta_STR1[i + 2, 0:2] = t1t2[1]
    Theta_STR1[i + 3, 0:2] = t1t2[1]

# Paso 3: Theta4 y Theta5 por PK2
noapHst0ip = noap @ np.linalg.inv(Hst0) @ np.append(pp, 1)
for i in range(0, 8, 2):
    pk2pt = np.linalg.inv(expScrew(np.append(Twist[:, 0], Theta_STR1[i, 0]))) @ noapHst0ip
    pk2pt = np.linalg.inv(expScrew(np.append(Twist[:, 1], Theta_STR1[i, 1]))) @ pk2pt
    pk2pt = np.linalg.inv(expScrew(np.append(Twist[:, 2], Theta_STR1[i, 2]))) @ pk2pt
    pk2p = pk2pt[:3]
    t4t5 = paden_kahan_two(Twist[:, 3], Twist[:, 4], pp, pk2p)
    Theta_STR1[i:i+2, 3:5] = t4t5

# Paso 4: Theta6 por PK1
noapHst0io = noap @ np.linalg.inv(Hst0) @ np.append(pk, 1)
for i in range(8):
    pk2pt = noapHst0io
    for j in range(5):
        pk2pt = np.linalg.inv(expScrew(np.append(Twist[:, j], Theta_STR1[i, j]))) @ pk2pt
    pk3p = pk2pt[:3]
    Theta_STR1[i, 5] = paden_kahan_one(Twist[:, 5:6], po, pk3p)

# Mostrar resultados
#print("Theta_STR1 =\n", Theta_STR1)

tIK1 = round((time() - start_time) * 1000, 1)
print(f"Time to solve IK Screw Theory: {tIK1} ms")

# Paso 5: Validar resultados con FK
for i in range(Theta_STR1.shape[0]):
    TwMagi = np.vstack((Twist, Theta_STR1[i, :]))
    HstRi = forward_kinematics_poe(TwMagi)
    noapi = HstRi @ Hst0
    print(f"Solution {i + 1}:\n", noapi)
