import sympy as sp
from toolbox.dh_fk import fk_denavit_hartenberg
from jacobianos.jacobiano_simbolico_lung import jacobiano_simbolico_lung
from jacobianos.jacobiano_simbolico_lung_screws import jacobiano_lung_screws
# Número de grados de libertad
n = 6  # Puedes cambiarlo

# Variables simbólicas
theta = sp.symbols(f'theta1:{n+1}')   # theta1, theta2, ..., thetan
a = sp.symbols(f'a1:{n+1}')           # a1, a2, ..., an
alpha = sp.symbols(f'alpha1:{n+1}')  
d = sp.symbols(f'd1:{n+1}')  

d_theta = {'theta1': theta[0], 'theta2':theta[1], 'theta3':theta[2], 'theta4': theta[3], 'theta5': theta[4], 'theta6': theta[5]}
d_alpha = {'alpha1': sp.pi/2, 'alpha2':0, 'alpha3': 0, 'alpha4': -sp.pi/2, 'alpha5': sp.pi/2, 'alpha6': 0}
d_a = {'a1':0, 'a2':a[1], 'a3':a[2], 'a4': a[3], 'a5': 0, 'a6': 0}
d_d = {'d1': 0, 'd2':0, 'd3':0, 'd4': 0, 'd5': 0, 'd6': d[5]}

# Tipo de articulación: 'R' para revoluta, 'P' para prismática
joint_types = ['R']*n 
Tnn = fk_denavit_hartenberg(d_theta, d_alpha, d_a, d_d)

jacobiano = jacobiano_simbolico_lung(Tnn, d_theta, d_a, d_d, joint_types)
jacobiano_screws = jacobiano_lung_screws(Tnn, d_alpha, d_a, d_d, joint_types)

print(jacobiano)
print(jacobiano_screws)
