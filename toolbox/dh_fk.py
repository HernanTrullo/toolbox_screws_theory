import sympy as sp
import numpy as np


def  fk_denavit_hartenberg(d_theta, d_alpha, d_a, d_d):
    # Construcción de las transformadas homogéneas T_0i
    Tnn = {}
    for i in range(len(d_theta)+1):    
        Tnn[f'{i}{i}'] = sp.eye(4)
        
    for i in range(len(d_theta)):
        ct = sp.cos(d_theta[f'theta{i+1}'])
        st = sp.sin(d_theta[f'theta{i+1}'])
        ca = sp.cos(d_alpha[f'alpha{i+1}'])
        sa = sp.sin(d_alpha[f'alpha{i+1}'])
        
        # Matriz DH estándar
        A_i = sp.Matrix([
            [ct, -st * ca,  st * sa, d_a[f'a{i+1}'] * ct],
            [st,  ct * ca, -ct * sa, d_a[f'a{i+1}'] * st],
            [0,      sa,      ca,      d_d[f'd{i+1}']],
            [0,       0,       0,       1]
        ])
        Tnn[f'{i}{i+1}'] = A_i
        Tnn[f'{i+1}{i}'] = invertir_T(A_i)
    
    for j in range(len(d_theta)-1):
        for i in range(j+2, len(d_theta)+1):
            # Se calcula la matrix hacia adelante
            Tnn_ = sp.eye(4)
            for k in range (j, i):
                Tnn_ = Tnn_ @ Tnn[f'{k}{k+1}']
            Tnn[f'{j}{i}'] = Tnn_
            Tnn[f'{i}{j}'] = invertir_T(Tnn_)
    return Tnn

def invertir_T(RT):
    R = RT[:3, :3]
    p = RT[:3, 3]
    R_T = R.T
    p_inv = -R_T * p

    T_inv = sp.eye(4)
    T_inv[:3, :3] = R_T
    T_inv[:3, 3] = p_inv

    return T_inv