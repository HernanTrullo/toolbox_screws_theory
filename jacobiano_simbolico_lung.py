import sympy as sp
import numpy as np

# Número de grados de libertad
n = 3  # Puedes cambiarlo

# Variables simbólicas
theta = sp.symbols(f'theta1:{n+1}')   # theta1, theta2, ..., thetan
d = sp.symbols(f'd1:{n+1}')           # d1, d2, ..., dn
a = sp.symbols(f'a1:{n+1}')           # a1, a2, ..., an
alpha = sp.symbols(f'alpha1:{n+1}')   # alpha1, ..., alphan

alpha = sp.zeros(n)
d = sp.zeros(n)

# Tipo de articulación: 'R' para revoluta, 'P' para prismática
joint_types = ['R']*n 

# Inicialización
T = sp.eye(4)
Ts = []  # Lista para almacenar las matrices T0_i

# Construcción de las transformadas homogéneas
for i in range(n):
    ct = sp.cos(theta[i])
    st = sp.sin(theta[i])
    ca = sp.cos(alpha[i])
    sa = sp.sin(alpha[i])
    
    # Matriz DH estándar
    A_i = sp.Matrix([
        [ct, -st * ca,  st * sa, a[i] * ct],
        [st,  ct * ca, -ct * sa, a[i] * st],
        [0,      sa,      ca,      d[i]],
        [0,       0,       0,       1]
    ])
    
    T = T @ A_i
    Ts.append(T)

# Posición del efector final (origen del último sistema)
p_n = Ts[-1][0:3, 3]

# Jacobiano simbólico
J = []

for i in range(n):
    if i == 0:
        T_prev = sp.eye(4)
    else:
        T_prev = Ts[i-1]
    
    R_prev = T_prev[0:3, 0:3]
    p_prev = T_prev[0:3, 3]
    
    z_prev = R_prev @ sp.Matrix([0, 0, 1])
    
    # i-1P*n* = R_prev @ (i-1 r_i) + iP*n*, lo haremos hacia atrás:
    p_star = p_n - p_prev
    
    if joint_types[i] == 'R':
        v = z_prev.cross(p_star)
        J_col = v.col_join(z_prev)
    else:
        J_col = z_prev.col_join(sp.Matrix([0, 0, 0]))
    
    J.append(J_col)

# Jacobiano final
J_sym = sp.Matrix.hstack(*J)

# Mostrar resultado
J_symplify = sp.simplify(J_sym)
sp.pprint(J_symplify, use_unicode=True)

j_eval = J_sym.subs({'theta1': sp.pi/2, 'theta2':sp.pi/4, 'theta3':sp.pi/8, 'a1':2, 'a2':1, 'a3':0.5}).evalf(4)
print(j_eval.T)
print(j_eval@ np.array([[0.5], [0.5], [0.5]]))
