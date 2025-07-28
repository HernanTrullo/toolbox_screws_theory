import sympy as sp
import numpy as np

# Se va comenzar con la articulación #3
# En esta articulación las condiciones iniciales son j=3; s4 = [0,0,1]' y s_o4 [0,0,0]'

def jacobiano_lung_screws(Tnn, alpha, a, d, Joint):
    j = 4
    S = {f'{j+1}': sp.Matrix([[0], [0], [1]])}
    So = {f'{j+1}': sp.Matrix([[0], [0], [0]])}
    zii = sp.Matrix([[0], [0], [1]])
    
    
    # forward algoritm
    n = len(alpha)
    for i in range(j+1,n):
        R_ji = Tnn[f'{j}{i}'][:3, :3]
        r_ii = sp.Matrix([[a[f'a{i}']], 
                        [d[f'd{i}']*sp.sin(alpha[f'alpha{i}'])], 
                        [d[f'd{i}']*sp.cos(alpha[f'alpha{i}'])]])
        S[f'{i+1}'] = R_ji @ zii
        So[f'{i+1}'] = So[f'{i}'] + R_ji @ r_ii
        
    # backwad algoritm
    for i in range(j-1,-1,-1):
        R_ji = Tnn[f'{j}{i}'][:3, :3]
        r_ii1 = sp.Matrix([[a[f'a{i+1}']], 
                        [d[f'd{i+1}']*sp.sin(alpha[f'alpha{i+1}'])], 
                        [d[f'd{i+1}']*sp.cos(alpha[f'alpha{i+1}'])]])
        S[f'{i+1}'] = R_ji @ zii
        So[f'{i+1}'] = So[f'{i+2}'] - Tnn[f'{j}{i+1}'][:3, :3] @ r_ii1
        
    J = {}
    for i in range(1, n+1):
        if Joint[i-1] == 'R':
            J[f'{i}'] = sp.Matrix.vstack(S[f'{i}'], So[f'{i}'].cross(S[f'{i}']))
        else:
            J[f'{i}'] = sp.Matrix.vstack(sp.Matrix([[0], [0], [0]]), S[f'{i}'])
            
    return J