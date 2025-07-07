import numpy as np
from toolbox.paden_kahan import paden_kahan_one
from toolbox.screws_code import joint2twist, forward_kinematics_poe

# Paso 0: Inicialización
pp = np.array([0.1, 0.1, 0.1])
Mag = np.pi / 8

# Paso 1: Definición del twist
Axis1 = np.array([0, 1, 0])    # Eje de rotación
p1 = np.array([0, 0, 0])       # Punto por donde pasa el eje
JointType1 = 'rot'             # Tipo de articulación
Twist = joint2twist(Axis1, p1, JointType1)  

# Paso 2: Cinemática directa con Mag (ángulo dado)
TwMag1 = np.vstack((Twist, Mag))    # Concatenar twist con el ángulo
HstR1 = forward_kinematics_poe(TwMag1)  # Transformación homogénea
pp_hom = np.append(pp, 1)            # Punto homogéneo
pk1h = HstR1 @ pp_hom                # Aplicar transformación
pk1 = pk1h[:3]                       # Extraer coordenadas

# Paso 3: Cinemática inversa (resolver Subproblema 1)
Theta1 = paden_kahan_one(Twist, pp, pk1) 

# Paso 4: Verificación con Theta1
TwMag2 = np.vstack((Twist, Theta1)) 
HstR2 = forward_kinematics_poe(TwMag2)
pk2h = HstR2 @ pp_hom
pk2 = pk2h[:3]

# Paso 5: Comparación
print("pk1 =", pk1)
print("pk2 =", pk2)
print("Diferencia:", np.linalg.norm(pk1 - pk2))

