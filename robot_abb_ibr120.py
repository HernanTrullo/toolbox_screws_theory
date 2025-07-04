import numpy as np
from toolbox.screws_code import forward_kinematics_poe, joint2twist

# Número de grados de libertad
n = 6
# Generar magnitudes aleatorias theta1-theta6
Mag = np.ones(6)*np.pi/4

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
Joint = ['rot', 'rot', 'rot', 'rot', 'rot', 'rot']

# Ejes de las articulaciones
Axis = np.column_stack((AxisZ, AxisY, AxisY, AxisX, AxisY, -AxisZ))

# Calcular twists
Twist = np.zeros((6, n))
for i in range(n):
    Twist[:, i] = joint2twist(Axis[:, i], Point[:, i], Joint[i])
    

print(Twist)
#Aplicar Forward Kinematics POE
TwMag = np.vstack((Twist, Mag))  # Apila Twist (6xn) y Mag (1xn)
print(TwMag)
HstR = forward_kinematics_poe(TwMag)

# matriz homogénea del efector final
print("HstR (matriz homogénea final del efector):\n", HstR)