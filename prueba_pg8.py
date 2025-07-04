import numpy as np
from toolbox.pardos_gotor import pardos_gotor_eight
from toolbox.screws_code import joint2twist, forward_kinematics_poe
from scipy.spatial.transform import Rotation as R

# --- PARTE 1: Magnitudes de las rotaciones ---
Mag = [np.pi / 4, np.pi / 8, np.pi / 16]  # t1, t2, t3

# --- PARTE 2: Pose inicial Hp ---
eul_angles = [np.pi / 4, np.pi / 4, np.pi / 5]  # ZYX
Rp = R.from_euler('ZYX', eul_angles).as_matrix()
Pp = np.array([np.pi / 2, np.pi / 3, np.pi / 3])
Hp = np.eye(4)
Hp[:3, :3] = Rp
Hp[:3, 3] = Pp

# --- PARTE 3: Ejes y puntos de los tornillos ---
r1 = np.array([5, 0, 0])
r2 = np.array([3, 0, 0])
r3 = np.array([1, 0, 0])
Point = np.column_stack((r1, r2, r3))

AxisY = np.array([0, 1, 0])
Axis = np.column_stack((AxisY, AxisY, AxisY))
JointType = ['rot', 'rot', 'rot']  # no usado en este ejemplo, solo para claridad

# --- PARTE 4: Construcción de los TWISTS ---
Twist = joint2twist(Axis[:, 0], Point[:, 0], 'rot')
for i in range(1, Point.shape[1]):
    Twist = np.column_stack((Twist, joint2twist(Axis[:, i], Point[:, i], 'rot')))

# --- PARTE 5: Cinemática directa para encontrar Hk ---
TwMag1 = np.vstack((Twist, Mag))  # (6, 3) + (1, 3) → (7, 3)
Hk = forward_kinematics_poe(TwMag1) @ Hp

# --- PARTE 6: Resolver la cinemática inversa con PG8 ---
t123 = pardos_gotor_eight(Twist[:, 0], Twist[:, 1], Twist[:, 2], Hp, Hk)
print("Soluciones t123:\n", t123)

# --- PARTE 7: Verificar cada solución aplicando FK ---
for i in range(t123.shape[0]):
    TwMagi = np.vstack((Twist, t123[i, :]))  # Combinar twist con ángulos calculados
    Hki = forward_kinematics_poe(TwMagi) @ Hp
    print(f"\nSolución {i+1}:")
    print(Hki)
