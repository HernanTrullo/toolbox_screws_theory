import sympy as sp
import numpy as np


def jacobiano_simbolico_lung(Tnn, d_theta, d_a, d_d, joint_types):
    # incialización de las matrices de transformación
    """
        Calculates the symbolic geometric Jacobian for a serial manipulator robot.

        Parameters:
        - d_theta: dictionary with the symbolic values of the angles theta{i}
        - d_d: dictionary with the displacements d{i}
        - d_a: dictionary with the lengths of the links a{i}
        - Tnn: dictionary with the transformation matrices T_{0}^{i} (SymPy.Matrix)
            - joint_types: list of ‘R’ (rotational) or ‘P’ (prismatic) for each joint

        Returns:
        - J: dictionary with the columns of the Jacobian J{i} for each joint i
    """
    J = {}
    n = len(d_theta)
    Pnn = {}
    Pnn[f'{n}{n}'] = sp.zeros(3,1)

    for i in range(n, 0, -1):
        R_0i = Tnn[f'0{i-1}'][:3,:3]
        z_ii = R_0i @ np.array([0, 0, 1]).transpose()
        
        if joint_types[i-1] == 'R':
            r_ii = np.array([[d_a[f'a{i}']*sp.cos(d_theta[f'theta{i}'])], [d_a[f'a{i}']*sp.sin(d_theta[f'theta{i}'])], [d_d[f'd{i}']]])
            P_in_ast = sp.simplify(R_0i @ r_ii + Pnn[f'{i}{n}'])
            Pnn[f'{i-1}{n}'] = P_in_ast
            J_ = np.vstack([sp.Matrix(z_ii).cross(P_in_ast), z_ii.reshape(3,1)])
            J[f'{i}'] = J_
        else:
            J[f'{i}'] = [z_ii, sp.zeros(3,1)]

    return J
