import numpy as np
from time import time
from toolbox.screws_code import joint2twist, forward_kinematics_poe, expScrew
from toolbox.paden_kahan import paden_kahan_two, paden_kahan_one
from toolbox.pardos_gotor import pardos_gotor_one, pardos_gotor_three, pardos_gotor_eight
import matplotlib.pyplot as plt
# Número de GDL
n = 4
D3 = 0.45
# Características mecánicas del UR16e
po = np.array([0, 0, 0])
pf = np.array([0, -D3, 0])
pp = np.array([0, -D3, D3])

AxisX = np.array([1, 0, 0])
AxisY = np.array([0, 1, 0])
AxisZ = np.array([0, 0, 1])
Point = np.column_stack([po, po, pf, pf])
JointType = ['rot'] * n
JointType[3] = 'tra'
Axis = np.column_stack([AxisZ, AxisY, -AxisY, AxisZ])

Twist = joint2twist(Axis[:, 0], Point[:, 0], JointType[0])
# Recorrer las demás articulaciones y acumular sus twists
for i in range(1, n):
    twist_i = joint2twist(Axis[:, i], Point[:, i], JointType[i])
    Twist = np.column_stack((Twist, twist_i)) 

# Matriz de transformación Hst0
Hst0 = np.array([[1, 0, 0, 0],[0,1,0, -D3],[0,0, 1, D3],[0,0,0,1]])
# Se aplica pardos-gotor 1 para hallar theta4 (tra)
# Trajectory generation
number_points = 1000
t = np.linspace(0, 2*np.pi, number_points)

# Trayectorias suaves para θ1–θ4
theta1_traj = 0.5 * np.sin(t)
theta2_traj = 0.5 * np.cos(t)
theta3_traj = 0.25 * np.sin(2*t)
theta4_traj = 0.1 + 0.05 * np.sin(t)  # Suponiendo θ4 es prismática

# Salida de desviaciones
desviacion_normal = np.zeros((number_points, 2))
soluciones_articulares = np.zeros((number_points, 5)) # 4 para las articuaciones y 1 para la clase de solucion
solucion_cartesiana_deseada_obtenida = np.zeros((number_points, 7)) # 1 sol deseada (x_des,y_des,z_des), 1 obtenida (x_ob, y_ob, z_ob) ,tipo de solucion (0,1)

for k in range (number_points):
    # Paso 1: Cinemática directa
    Mag = np.array([theta1_traj[k], theta2_traj[k], theta3_traj[k], theta4_traj[k]])  # Theta1–6 aleatorios
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
    noap_if = noap@np.linalg.inv(Hst0)@ np.append(pp, 1)
    # Se obtiene el punto pfp, de la multiplicacion del lado izquiero cte, pies
    for i in range(4):
        TwMag_4 = np.hstack((TwMag[:,3], ThSTR[i, 3])) # indica el angulo th4
        Hpp = expScrew(TwMag_4) @ np.append(pp,1)
        pfp = Hpp[:3]
        
        pkp = np.linalg.inv(expScrew(np.hstack((Twist[:, 0], ThSTR[i, 0])))) @ noap_if
        pkp = np.linalg.inv(expScrew(np.hstack((Twist[:, 1], ThSTR[i, 1])))) @ pkp
        
        ThSTR[i, 2] = paden_kahan_one(Twist[:, 2:3], pfp, pkp[:3])
    
    # Paso 3: Verificación con FK
    for i in range(4):
        TwMagi = np.vstack((Twist[:, :4], ThSTR[i, :4]))
        HstRi = forward_kinematics_poe(TwMagi)
        noapi = HstRi @ Hst0
        delta_normal = np.linalg.norm(noap[:3, 3]-noapi[:3,3])
        if delta_normal < 1e-13:
            desviacion_normal[k] = [delta_normal, i]
            soluciones_articulares[k] = np.hstack([ThSTR[i], i])
            solucion_cartesiana_deseada_obtenida[k] = np.hstack([noap[:3, 3],noapi[:3,3], i]).reshape(1, -1)
            break
        
data = desviacion_normal
valores = data[:, 0]
clases = data[:, 1]
indices = np.arange(len(valores))

# Crear figura
plt.figure(figsize=(10, 5))
# Línea continua (sin importar la clase)
plt.plot(indices, valores, color='gray', linewidth=1, alpha=0.5, label='Error')
# Puntos coloreados según clase
plt.scatter(indices[clases == 0], valores[clases == 0], color='red', label='Solucion 1', s=10)
plt.scatter(indices[clases == 1], valores[clases == 1], color='blue', label='Solucion 2', s=10)
# Estética
plt.title('Gráfico con línea y puntos por clase')
plt.xlabel('Índice')
plt.ylabel('Valor')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

matriz = soluciones_articulares
# Crear la figura y los ejes (uno por articulación)
fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
nombres = [r'$\theta_1$', r'$\theta_2$', r'$\theta_3$', r'$\theta_4$']

# Extraer los índices por tipo de solución
indices_tipo_0 = matriz[:, 4] == 0
indices_tipo_1 = matriz[:, 4] == 1

# Vector de tiempo o índice
x = np.arange(matriz.shape[0])

# Graficar cada articulación
for i in range(4):
    axs[i].plot(x[indices_tipo_0], matriz[indices_tipo_0, i], '.', color='blue', label='Solucion 0')
    axs[i].plot(x[indices_tipo_1], matriz[indices_tipo_1, i], '+', color='red', label='Solucion 1')
    axs[i].set_ylabel(nombres[i])
    axs[i].grid(True)
    if i == 0:
        axs[i].legend()

axs[-1].set_xlabel('Muestra')
plt.suptitle('Trayectorias de las articulaciones por tipo de solución')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# -------------------------------------------------------------------
# Plotear datos cartesianos
matriz = solucion_cartesiana_deseada_obtenida
# Separar los datos
x_des, y_des, z_des = matriz[:, 0], matriz[:, 1], matriz[:, 2]
x_ob, y_ob, z_ob = matriz[:, 3], matriz[:, 4], matriz[:, 5]
tipo = matriz[:, 6]

# Crear figura 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Graficar las posiciones deseadas (gris claro)
ax.scatter(x_des, y_des, z_des, c='gray', marker='o', label='Deseada', alpha=0.4)

# Separar por tipo de solución
idx_tipo_0 = tipo == 0
idx_tipo_1 = tipo == 1

# Tipo 0: puntos azules
ax.scatter(x_ob[idx_tipo_0], y_ob[idx_tipo_0], z_ob[idx_tipo_0],
           c='blue', marker='.', label='Obtenida (tipo 0)')

# Tipo 1: cruces rojas
ax.scatter(x_ob[idx_tipo_1], y_ob[idx_tipo_1], z_ob[idx_tipo_1],
           c='red', marker='+', label='Obtenida (tipo 1)')

# Configuración del gráfico
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Comparación 3D: Posiciones Deseadas vs Obtenidas')
ax.legend()
plt.tight_layout()
plt.show()


