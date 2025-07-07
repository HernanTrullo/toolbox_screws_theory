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
from toolbox.pardos_gotor import pardos_gotor_five
from toolbox.screws_code import joint2twist, forward_kinematics_poe

# Paso 0: Inicialización
pp = np.array([1, 2, 3])
Mag = np.pi / 8

# Paso 1: Definición del twist
Axis1 = np.array([0, 1, 0])    # Eje de rotación
p1 = np.array([0, 0, 0])       # Punto por donde pasa el eje
JointType1 = 'rot'             # Tipo de articulación
Twist = joint2twist(Axis1, p1, JointType1)  # Función asumida

# Paso 2: Cinemática directa con Mag (ángulo dado)
TwMag1 = np.vstack((Twist, Mag))    # Concatenar twist con el ángulo
HstR1 = forward_kinematics_poe(TwMag1)  # Transformación homogénea
pp_hom = np.append(pp, 1)            # Punto homogéneo
pk1h = HstR1 @ pp_hom                # Aplicar transformación
pk1 = pk1h[:3]                       # Extraer coordenadas

# Paso 3: Cinemática inversa (resolver Subproblema 1)
Theta1 = pardos_gotor_five(Twist, pp, pk1)  # Función asumida

# Paso 4: Verificación con Theta1
TwMag2 = np.vstack((Twist, Theta1[0])) 
HstR2 = forward_kinematics_poe(TwMag2)
pk2h = HstR2 @ pp_hom
pk2 = pk2h[:3]

# Paso 5: Comparación
print("pk1 =", pk1)
print("pk2 =", pk2)
print("Angulo:", Theta1[0] - Theta1[1])

