import numpy as np
from toolbox.screws_code import tform2eul, geo_jacobian_s, joint2twist, forward_kinematics_poe, axis2skew


# ----------------------
# Datos del robot (posición de referencia)
# ----------------------
po = np.array([0, 0, 0])
pu = np.array([1088, 2500, 0])
pk = np.array([1.468, 2.500, 0])
pr = np.array([2.443, 2.500, 0])
pf = np.array([2.643, 1.613, 0])
pp = np.array([3.000, 1.613, 0])

AxisX = np.array([1, 0, 0])
AxisY = np.array([0, 1, 0])
AxisZ = np.array([0, 0, 1])

Point = np.column_stack([pu, pk, pr, pf, pf, pf])
Joint = ['tra', 'rot', 'rot', 'rot', 'rot', 'rot']
Axis = np.column_stack([AxisZ, -AxisZ, -AxisZ, -AxisY, -AxisZ, AxisX])

# ----------------------
# Twist de cada articulación
# ----------------------

# Inicializar el primer twist
Twist = joint2twist(Axis[:, 0], Point[:, 0], Joint[0])
n = len(Joint)
# Recorrer las demás articulaciones y acumular sus twists
for i in range(1, n):
    twist_i = joint2twist(Axis[:, i], Point[:, i], Joint[i])
    Twist = np.column_stack((Twist, twist_i)) 

# ----------------------
# Pose home de la herramienta (matriz homogénea)
# ----------------------
Hst0 = np.array([
    [0, 0, 1, 3],
    [-1, 0, 0, 1.6130],
    [0, -1, 0, 0],
    [0, 0, 0, 1]
])

# ----------------------
# Cinemática directa con ángulos en grados
# ----------------------
Theta = np.array([100, 100, 100, 100, 100, 100])
TwMag = np.vstack([Twist, Theta * np.pi / 180])
noap = forward_kinematics_poe(TwMag) @ Hst0

# PLa posision objetivo que se desea alcanzar
t1goal = np.hstack([noap[0:3, 3], tform2eul(noap, 'XYZ')])
print("t1goal =", t1goal)

# ----------------------
# Velocidad deseada en el marco espacial (espacial)
# ----------------------
VtSr = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

# ----------------------
# Jacobiano geométrico espacial
# ----------------------
JstS = geo_jacobian_s(TwMag)

# Inversa diferencial: velocidades articulares
omega_skew = axis2skew(VtSr[3:6])
translation_part = VtSr[0:3] - omega_skew @ t1goal[0:3]
Thetap = np.linalg.solve(JstS, np.hstack([translation_part, VtSr[3:6]]))
print("Thetap =", Thetap)

# ----------------------
# Directa diferencial: velocidad cartesiana
# ----------------------
VstS = JstS @ Thetap
VtS = np.hstack([VstS[0:3] + axis2skew(VstS[3:6]) @ t1goal[0:3], VstS[3:6]])
print("VtS =", VtS)
