import numpy as np
from time import time
from toolbox.screws_code import joint2twist, forward_kinematics_poe, expScrew
from toolbox.paden_kahan import paden_kahan_one
from toolbox.pardos_gotor import pardos_gotor_five, pardos_gotor_three, pardos_gotor_eight

# Número de GDL
n = 6
Mag = np.array([np.random.rand()*np.pi for _ in range(n)])  # Theta1–6 aleatorios

# Características mecánicas del UR5
po = np.array([0, 0, 0])
pk = np.array([0, 0, 0.1625])
pr = np.array([0.425, 0, 0.1625])
pf = np.array([0.817, 0.1333, 0.1625])
pg = np.array([0.817, 0.1333, 0.0628])
pp = np.array([0.817, 0.2333, 0.0628])
AxisX = np.array([1, 0, 0])
AxisY = np.array([0, 1, 0])
AxisZ = np.array([0, 0, 1])
Point = np.column_stack([po, pk, pr, pf, pg, pp])
JointType = ['rot'] * 6
Axis = np.column_stack([AxisZ, AxisY, AxisY, AxisY, -AxisZ, AxisY])

Twist = joint2twist(Axis[:, 0], Point[:, 0], JointType[0])
n = len(JointType)
# Recorrer las demás articulaciones y acumular sus twists
for i in range(1, n):
    twist_i = joint2twist(Axis[:, i], Point[:, i], JointType[i])
    Twist = np.column_stack((Twist, twist_i)) 
# Matriz de transformación Hst0
Hst0 = np.array([[-1, 0, 0, 0.817],[0,0,1, 0.2333],[0,1, 0, 0.0628],[0,0,0,1]])

def ik_robot_ur(HstR, num_soluition = 0):
    # Paso 1: Cinemática directa
    noap = HstR

    # Paso 2: Cinemática inversa
    Theta_STR4 = np.zeros((8, n))

    # Paso 2.1: Theta1 usando PG5
    noapHst0ig = noap @ np.linalg.inv(Hst0) @ np.append(pg, 1)
    pk1 = noapHst0ig[:3]
    t1 = pardos_gotor_five(Twist[:, 0:1], pg, pk1)

    w1 = Twist[3:, 0]
    v1 = Twist[:3, 0]
    r1 = np.cross(w1, v1) / np.linalg.norm(w1)**2
    v = pk1 - r1
    vw1 = np.outer(w1, w1) @ v
    vp1 = v - vw1
    nvp = np.linalg.norm(vp1)

    u = pg - r1
    uw1 = np.outer(w1, w1) @ u
    up1 = u - uw1
    nup = np.linalg.norm(up1)

    t11 = t1[0] - np.arcsin(pg[1]/nvp) + np.arcsin(pg[1]/nup)
    t12 = t1[1] + np.arcsin(pg[1]/nvp) + np.arcsin(pg[1]/nup)

    Theta_STR4[0:4, 0] = np.real(t11)
    Theta_STR4[4:8, 0] = np.real(t12)
    # Paso 2.2: Theta5 usando PG3 y PK1
    noapHst0ip = noap @ np.linalg.inv(Hst0) @ np.append(pp, 1)
    for i in range(0, 6, 4):
        pk2ph = np.linalg.inv(expScrew(np.append(Twist[:, 0], Theta_STR4[i, 0]))) @ noapHst0ip
        pk2p = pk2ph[:3]
        w7 = np.array([1, 0, 0])
        x7 = np.hstack((w7.reshape(-1, 1), np.zeros((3,1))))
        t7 = pardos_gotor_three(x7, np.append(pk2p[:2], 0), np.append(pg[:2], 0), np.linalg.norm(pp - pg))
        pk2 = pk2p + w7 * t7[1]
        t51 = paden_kahan_one(Twist[:, 4:5], pp, pk2)
        Theta_STR4[i:i+2, 4] = np.real(t51)
        Theta_STR4[i+2:i+4, 4] = np.real(-t51)

    # Paso 2.3: Theta6 (geometría)
    ox, oy = noap[0, 1], noap[1, 1]
    nx, ny = noap[0, 0], noap[1, 0]
    for i in range(0, 8, 2):
        s1 = np.sin(Theta_STR4[i, 0])
        c1 = np.cos(Theta_STR4[i, 0])
        s5 = np.sin(Theta_STR4[i, 4])
        t61a = np.arctan2((ox*s1 - oy*c1)/s5, (ny*c1 - nx*s1)/s5)
        Theta_STR4[i:i+2, 5] = np.real(t61a)

    # Paso 2.4: Theta2–4 usando PG8
    for i in range(0, 8, 2):
        Hp = expScrew(np.append(Twist[:, 5], Theta_STR4[i, 5])) @ Hst0
        Hp = expScrew(np.append(Twist[:, 4], Theta_STR4[i, 4])) @ Hp
        Hk = np.linalg.inv(expScrew(np.append(Twist[:, 0], Theta_STR4[i, 0]))) @ noap
        t234 = pardos_gotor_eight(Twist[:, 1], Twist[:, 2], Twist[:, 3], Hp, Hk)
        Theta_STR4[i:i+2, 1:4] = t234


    """# Paso 3: Comprobación con cinemática directa
    for i in range(Theta_STR4.shape[0]):
        TwMagi = np.vstack((Twist, Theta_STR4[i, :]))
        noapi = forward_kinematics_poe(TwMagi) @ Hst0
        print(f"Pose solución {i+1}:\n", noapi)"""
        
    return Theta_STR4[num_soluition, :]
