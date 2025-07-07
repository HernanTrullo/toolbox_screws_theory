# Copyright (C) 2025 Hernan Trullo
#
# This file is part of [screws theory toolbox].
#
# [screws ik] is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# [screws ik] is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import numpy as np
from toolbox.pardos_gotor import pardos_gotor_three
from toolbox.screws_code import joint2twist, forward_kinematics_poe

# Datos de entrada
pp = np.array([2, 3, 4])  # Punto inicial
pk = np.array([5, 4, 6])  # Punto final
Mag = np.pi / 5           # Magnitud de prueba

# Definición del tornillo (twist)
Axis1 = np.array([1, 0, 0])
p1 = np.array([0, 0, 0])
JointType1 = 'tra'

# Obtener el twist
Twist = joint2twist(Axis1, p1, JointType1)  # Se espera un vector de 6x1

# Paso 1: Cinemática directa con una magnitud cualquiera
TwMag1 = np.vstack((Twist, Mag))
HstR1 = forward_kinematics_poe(TwMag1)  # Se espera una matriz homogénea 4x4
pc1h = HstR1 @ np.append(pp, 1)
pc1 = pc1h[:3]
de1 = np.linalg.norm(pk - pc1)
print(f"de1 = {de1}")

# Paso 2: Resolver el subproblema PK3
Theta1 = pardos_gotor_three(Twist, pp, pk, de1)  # Se espera un array de 2 soluciones
print(f"Theta1 = {Theta1}")

# Paso 3: Validar la solución con ambas magnitudes encontradas
TwMag2 = np.vstack((Twist, [Theta1[0]]))
HstR2 = forward_kinematics_poe(TwMag2)
pc2h = HstR2 @ np.append(pp, 1)
pc2 = pc2h[:3]
de2 = np.linalg.norm(pk - pc2)
print(f"de2 = {de2}")

TwMag3 = np.vstack((Twist, [Theta1[1]]))
HstR3 = forward_kinematics_poe(TwMag3)
pe3h = HstR3 @ np.append(pp, 1)
pe3 = pe3h[:3]
de3 = np.linalg.norm(pk - pe3)
print(f"de3 = {de3}")