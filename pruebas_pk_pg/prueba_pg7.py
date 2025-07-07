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