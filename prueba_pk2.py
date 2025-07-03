import numpy as np
from toolbox.paden_kahan import paden_kahan_two
from toolbox.screws_code import forward_kinematics_poe, joint2twist

# Paso 0: Inicialización
pp = np.array([0.3, 0.3, 0.3])
Mag = np.array([0.1, np.pi/3])

# Paso 1: Definición del twist
Axis = np.array([[1, 0, 0], [0, 1, 0]]).transpose()   # Ejes de rotacion
Point = np.array([[1,0,0],[0,1,0]]).transpose()
JointType = np.array(['rot', 'rot'])       # Tipo de articulación
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
pk1 = pk1h[:3]                              # Extraer coordenadas

# Calcular la solución de cinemática inversa
Th1Th2 = paden_kahan_two(Twist[:, 0], Twist[:, 1], pp, pk1)

# Paso 3: Validar ambas soluciones aplicando cinemática directa
TwMag2 = np.vstack((Twist, Th1Th2[0]))
HstR2 = forward_kinematics_poe(TwMag2)
pk2h = HstR2 @ np.append(pp, 1)
pk2 = pk2h[:3]

TwMag3 = np.vstack((Twist, Th1Th2[1]))
HstR3 = forward_kinematics_poe(TwMag3)
pk3h = HstR3 @ np.append(pp, 1)
pk3 = pk3h[:3]

# Imprimir para comprobar que pk1 ≈ pk2 ≈ pk3
print("pk1:", pk1)
print("pk2:", pk2)
print("pk3:", pk3)
print(Mag)
print(Th1Th2)

