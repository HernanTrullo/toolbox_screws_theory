import numpy as np
from time import time
from toolbox.screws_code import joint2twist, forward_kinematics_poe, expScrew
from toolbox.paden_kahan import paden_kahan_two, paden_kahan_one
from toolbox.pardos_gotor import pardos_gotor_one, pardos_gotor_three, pardos_gotor_eight

# Número de GDL
n = 4
Mag = np.array([np.pi/4, np.pi/4, np.pi/4, 0.4])  # Theta1–6 aleatorios

D3 = 0.45
# Características mecánicas del UR16e
po = np.array([0, 0, 0])
pf = np.array([0, -D3, 0])
pp = np.array([0, -2*D3, 0])

AxisX = np.array([1, 0, 0])
AxisY = np.array([0, 1, 0])
AxisZ = np.array([0, 0, 1])
Point = np.column_stack([po, po, pf, pf])
JointType = ['rot'] * n
JointType[3] = 'tra'
Axis = np.column_stack([AxisZ, AxisY, -AxisY, AxisX])

Twist = joint2twist(Axis[:, 0], Point[:, 0], JointType[0])
# Recorrer las demás articulaciones y acumular sus twists
for i in range(1, n):
    twist_i = joint2twist(Axis[:, i], Point[:, i], JointType[i])
    Twist = np.column_stack((Twist, twist_i)) 

# Matriz de transformación Hst0
Hst0 = np.array([[1, 0, 0, 0],[0,1,0, -D3],[0,0, 1, 0],[0,0,0,1]])
# Se aplica pardos-gotor 1 para hallar theta4 (tra)

# Paso 1: Cinemática directa
TwMag = np.vstack((Twist, Mag))
noap = forward_kinematics_poe(TwMag) @ Hst0

#cinematica inversa
#Hallar theta4 y theta5 hallando tetha4 primero, asumiendo que theta5 no existe
# El angulo lo dividimos entre las dos articulaciones.
# Aplicamos el punto pf a ambos lados de la ecuación y simplificamos las primeras tres articulaciones
# t1 y t2 no afectan po y th3 no afecta pf, con lo cual de = norm(po-pf)


ThSTR = np.zeros((4, 4))
pkf = noap@np.linalg.inv(Hst0)@ np.append(pf, 1)
Theta4 = pardos_gotor_three(Twist[:, 3:4], pf, po, np.linalg.norm(pkf[0:3] - po))
ThSTR[0:2, 3] = Theta4[0]
ThSTR[2:4, 3] = Theta4[1]

# Ahora se halla para theta 1 y 2, con pk1 simplification y pk2 sobproblem

pkf = noap@np.linalg.inv(Hst0)@ np.append(pf, 1)
Th1Th2 = []
for i in range(2):
    TwMag_4 = np.hstack((TwMag[:,3], Theta4[i]))
    Hpf = expScrew(TwMag_4) @ np.append(pf,1)
    pff = Hpf[:3]
    Th1Th2 = paden_kahan_two(Twist[:, 0], Twist[:, 1], pff, pkf[:3])
    ThSTR[2*i, 0:2] = Th1Th2[0]
    ThSTR[2*i+1, 0:2] = Th1Th2[1]


# Ahora la solucion para the3 meidante pk1, pues se pasan los exp^-1 th1, th2 a premultiplicar
noap_if = noap@np.linalg.inv(Hst0)@ np.append(po, 1)
# Se obtiene el punto pfp, de la multiplicacion del lado izquiero cte, pies
for i in range(4):
    TwMag_4 = np.hstack((TwMag[:,3], ThSTR[i, 3])) # indica el angulo th4
    Hpp = expScrew(TwMag_4) @ np.append(po,1)
    pfp = Hpp[:3]
    
    pkp = np.linalg.inv(expScrew(np.hstack((Twist[:, 0], ThSTR[i, 0])))) @ noap_if
    pkp = np.linalg.inv(expScrew(np.hstack((Twist[:, 1], ThSTR[i, 1])))) @ pkp
    
    ThSTR[i, 2] = paden_kahan_one(Twist[:, 2:3], pfp, pkp[:3])

print("FK Noap \n", noap)
# Paso 3: Verificación con FK
for i in range(4):
    TwMagi = np.vstack((Twist[:, :4], ThSTR[i, :4]))
    HstRi = forward_kinematics_poe(TwMagi)
    noapi = HstRi @ Hst0
    print(f"FK result STR1, solution {i+1}:\n", noapi)
    print(np.linalg.norm(noap-noapi))

print()
