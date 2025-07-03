import numpy as np
from toolbox.pardos_gotor import pardos_gotor_one
from toolbox.screws_code import joint2twist, forward_kinematics_poe

# Datos de entrada
pp = np.array([2, 3, 4])  # Punto inicial
Mag = np.pi / 5           # Magnitud de prueba

# Definición del tornillo (twist)
Axis1 = np.array([0, 1, 0])
p1 = np.array([1, 2, 3])
JointType1 = 'tra'

# Obtener el twist
Twist = joint2twist(Axis1, p1, JointType1)  # Se espera un vector de 6x1

# Paso 1: Cinemática directa con una magnitud cualquiera
TwMag1 = np.vstack((Twist, Mag))
HstR1 = forward_kinematics_poe(TwMag1)  # Se espera una matriz homogénea 4x4
pk1h = HstR1 @ np.append(pp, 1)
pk1 = pk1h[:3]
print(f"pk1 = {pk1}")

# Paso 2: Resolver el subproblema PK3
Theta1 = pardos_gotor_one(Twist, pp, pk1) # se espera un valor
print(f"Theta1 = {Theta1}")

# Paso 3: Validar la solución con ambas magnitudes encontradas
TwMag2 = np.vstack((Twist, Theta1))
HstR2 = forward_kinematics_poe(TwMag2)
pk2h = HstR2 @ np.append(pp, 1)
pk2 = pk2h[:3]
print(f"pk2 = {pk2}")
