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
from toolbox.pardos_gotor import pardos_gotor_seven
from toolbox.screws_code import joint2twist, forward_kinematics_poe

# Paso 0: Inicialización
pp = np.array([0.6,0.7,0.8])
Mag = np.array([np.pi/2, np.pi/4, np.pi/8])

# Paso 1: Definición del twist
Axis = np.array([[0, 0, 1], [0, 1, 0],[0, 1, 0]]).transpose()   # Ejes de rotacion
Point = np.array([[0,0,0],[1,0,1], [1,0,2]]).transpose()
JointType = np.array(['rot', 'rot', 'rot'])       # Tipo de articulación
n = Axis.shape[1]  # Número de articulaciones

# Inicializar el primer twist
Twist = joint2twist(Axis[:, 0], Point[:, 0], JointType[0])
# Recorrer las demás articulaciones y acumular sus twists
for i in range(1, n):
    twist_i = joint2twist(Axis[:, i], Point[:, i], JointType[i])
    Twist = np.column_stack((Twist, twist_i)) 

# Paso 2: Cinemática directa con Mag (ángulo dado)
TwMag1 = np.vstack((Twist, Mag))            # Concatenar twist con el ángulo
HstR1 = forward_kinematics_poe(TwMag1)      # Transformación homogénea
pp_hom = np.append(pp, 1)                   # Punto homogéneo
pk1h = HstR1 @ pp_hom                       # Aplicar transformación
pk1 = pk1h[:3]                            # Extraer coordenadas

# Paso 3: Cinemática inversa (resolver Subproblema 1)
Theta1 = pardos_gotor_seven(Twist[:,0], Twist[:,1], Twist[:,2], pp, pk1)  

# Paso 4: Verificación con Theta1
TwMag2 = np.vstack((Twist, Theta1[0])) 
HstR2 = forward_kinematics_poe(TwMag2)
pk2h = HstR2 @ pp_hom
pk2 = pk2h[:3]

# Paso 5: Comparación
print("pk1 =", pk1)
print("pk2 =", pk2)
print(Theta1)