# Copyright (C) 2025 Hernan Trullo
#
# This file is part of [Nombre de tu proyecto o biblioteca].
#
# [Nombre de tu proyecto o biblioteca] is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# [Nombre de tu proyecto o biblioteca] is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import numpy as np
from toolbox.screws_code import intersect_lines_3d, joint2twist
from toolbox.paden_kahan import paden_kahan_one

def pardos_gotor_one(x1, pp, pk):
    """
    Compute the angle theta to traslate point `pp` to `pk`
    around the screw axis defined by twist vector `x1`, assuming pure traslation.

    Parameters:
    - x1: (6,) ndarray, the twist [v; w], where v and w are (3,)
    - pp: (3,) ndarray, the initial position
    - pk: (3,) ndarray, the target position

    Returns:
    - theta1: float, the angle of traslation
    """
    v1 = x1[0:3, 0]
    return np.dot(v1, pk-pp)

def pardos_gotor_two(x1, x2, pp, pk):
    """
    Calculates angles theta1 and theta2 according to the Pardos-Gotor subproblem 2.

    Parameters:
    - x1, x2: 6D vectors where the first 3 elements are direction vectors.
    - pp, pk: points in 3D space.

    Returns:
    - Theta1Theta2: list of values (t11, t12)
    """
    v1 = x1[:3]
    v2 = x2[:3]
    
    pc = intersect_lines_3d(np.column_stack((v1, v2)), np.column_stack((pk, pp)))

    if np.linalg.norm(pc) == np.inf:
        t11 = 0
        t21 = 0
    else:
        t21 = np.dot(v2, pc - pp)
        t11 = np.dot(v1, pk - pc)

    return [t11, t21]

def pardos_gotor_three(x1, pp, pk, de):
    """
    Pardos-Gotor Subproblem PG3:
    Find the parameters t1 such that the line defined by pp and direction x1
    intersects a sphere centered at pk and radius de.

    Parameters:
    - x1: input vector (using x1[0:3] as the line direction)
    - pp: start point of the line
    - pk: center of the sphere
    - de: radius of the sphere

    Returns:
    - Theta1: possible solutions for parameter t1 (array of two values)
    """
    v1 = x1[:3, 0]
    kmp = pk - pp
    kmpp = np.dot(v1, kmp)

    # Primera aproximación
    t11 = kmpp
    t12 = kmpp

    root = np.real(kmpp**2 - np.linalg.norm(kmp)**2 + de**2)

    if root > 0:
        sqrt_root = np.sqrt(root)
        t11 += sqrt_root
        t12 -= sqrt_root

    Theta1 = np.array([t11, t12])
    return Theta1

def pardos_gotor_four(x1, x2, pp, pk):
    """
    Calculates the inverse kinematic solutions for two consecutive twists
    whose axes are parallel but do not coincide spatially. This problem is solved
    in the SE(3) transformation group.

    Parameters:
    ----------
    x1 :(6,), Twist corresponding to the second twist applied (applied afterwards).
        Format: [v1; w1] where v1, w1 are vectors of dimension (3,).
    
    x2 :(6,), Twist corresponding to the first applied twist (applied first).
        Format: [v2; w2] where v2, w2 are vectors of dimension (3,).
    
    pp :(3,), Starting point (p) in space.
    
    pk :(3,), End point (k) in space.

    Returns:
    -------
    Theta1Theta2 :(2, 2), Matrix with two possible pairs of solutions [θ1, θ2] for the rotations.
        Each row represents a double solution to the problem.
"""    
    # Descomposición de los twists
    v1 = x1[:3]
    w1 = x1[3:]
    v2 = x2[:3]
    w2 = x2[3:]
    
    if not np.allclose(np.cross(w1, w2), [0, 0, 0]):
        raise NoParalelVectors("Los vectores deben ser paralelos")

    # Cálculo de los puntos de intersección de ejes
    r1 = np.cross(w1, v1) / np.linalg.norm(w1)**2
    r2 = np.cross(w2, v2) / np.linalg.norm(w2)**2

    # Proyección de pk y pp sobre los ejes
    v = pk - r1
    vw1 = np.outer(w1, w1) @ v
    vp = v - vw1
    nvp = np.linalg.norm(vp)
    c1 = r1 + vw1

    u = pp - r2
    uw2 = np.outer(w2, w2) @ u
    up = u - uw2
    nup = np.linalg.norm(up)
    c2 = r2 + uw2

    # Intersección de circunferencias
    c2c1 = c2 - c1
    nc2c1 = np.linalg.norm(c2c1)
    wa = c2c1 / nc2c1
    wh = np.cross(w1, wa)

    if nc2c1 >= (nvp + nup) or nvp >= (nc2c1 + nup) or nup >= (nc2c1 + nvp):
        pc = c1 + nvp * wa
        pd = pc
    else:
        a = (nc2c1**2 - nup**2 + nvp**2) / (2 * nc2c1)
        h = np.sqrt(abs(nvp**2 - a**2))
        pc = c1 + a * wa + h * wh
        pd = c1 + a * wa - h * wh

    # Proyecciones ortogonales para solución con Paden-Kahan 1
    m1 = pc - r1
    m1p = m1 - (np.outer(w1, w1) @ m1)
    n1 = pd - r1
    n1p = n1 - (np.outer(w1, w1) @ n1)

    m2 = pc - r2
    m2p = m2 - (np.outer(w2, w2) @ m2)
    n2 = pd - r2
    n2p = n2 - (np.outer(w2, w2) @ n2)

    # Soluciones angulares
    t11 = np.arctan2(np.real(np.dot(w1, np.cross(m1p, vp))), np.real(np.dot(m1p, vp)))
    t12 = np.arctan2(np.real(np.dot(w1, np.cross(n1p, vp))), np.real(np.dot(n1p, vp)))
    t21 = np.arctan2(np.real(np.dot(w2, np.cross(up, m2p))), np.real(np.dot(up, m2p)))
    t22 = np.arctan2(np.real(np.dot(w2, np.cross(up, n2p))), np.real(np.dot(up, n2p)))

    Theta1Theta2 = np.array([[t11, t21],
                            [t12, t22]])
    return Theta1Theta2

def pardos_gotor_five(x1, pp, pk):
    """
    Compute the angle theta (Theta1) to rotate point `pp` to `pk`
    around the screw axis defined by twist vector `x1`, assuming pure rotation.

    Parameters:
    - x1: (6,) ndarray, the twist [v; w], where v and w are (3,)
    - pp: (3,) ndarray, the initial position
    - pk: (3,) ndarray, the target position

    Returns:
    - theta1: float, the angle (in radians) of rotation
    """
    v1 = x1[0:3,0]
    w1 = x1[3:,0]
    
    # Compute the point on the axis of rotation
    r1 = np.cross(w1, v1) / (np.linalg.norm(w1) ** 2)
    
    # Compute projections onto plane perpendicular to w1
    u = pp - r1
    up = u - np.dot(w1, np.dot(w1, u))  # w1 * (w1' * u)
    
    v = pk - r1
    vp = v - np.dot(w1, np.dot(w1, v))  # w1 * (w1' * v)

    # Compute theta using atan2
    theta11 = np.arctan2(np.dot(w1, np.cross(up, vp)), np.dot(up, vp))
    theta12 = -np.sign(theta11)*(np.pi - np.abs(theta11))
    
    return np.array([theta11, theta12])

def pardos_gotor_six(x1, x2, pp, pk):
    """
    Pardos-Gotor Subproblem Six (PG6) – Rotation around two skew axes applied to a point.
    
    This function computes the inverse kinematics (IK) of a point that moves through 
    two consecutive screw rotations with skew axes (i.e., non-parallel and non-intersecting).
    It returns up to two possible angle solutions (theta1, theta2) that transform point `pp`
    to point `pk` using the screw motions defined by twists `x2` followed by `x1`.
    
    Parameters:
        x1 (np.ndarray): 6D twist vector [v1; w1] for the first screw.
        x2 (np.ndarray): 6D twist vector [v2; w2] for the second screw.
        pp (np.ndarray): Initial 3D point after the first rotation.
        pk (np.ndarray): Final 3D point after the second rotation.
    
    Returns:
        np.ndarray: A 2x2 matrix where each row [theta1, theta2] is a valid IK solution.
    """
    
    v1, w1 = x1[:3], x1[3:]
    v2, w2 = x2[:3], x2[3:]
    
    r1 = np.cross(w1, v1) / np.linalg.norm(w1)**2
    r2 = np.cross(w2, v2) / np.linalg.norm(w2)**2
    
    v = pk - r1
    vw1 = w1 @ w1.T @ v if v.ndim == 2 else w1 * (w1 @ v)
    vp1 = v - vw1
    nvp = np.linalg.norm(vp1)
    o1 = r1 + vw1
    
    u = pp - r2
    uw2 = w2 @ w2.T @ u if u.ndim == 2 else w2 * (w2 @ u)
    up2 = u - uw2
    nup = np.linalg.norm(up2)
    o2 = r2 + uw2
    
    # Plane distances (normal form)
    d1 = w1 @ o1
    d2 = w2 @ o2
    
    # Line of intersection between the planes
    v3 = np.cross(w1, w2)
    w12 = w1 @ w2
    r3 = (w1 * (d1 - d2 * w12) + w2 * (d2 - d1 * w12)) / (1 - w12)
    x3 = np.vstack((v3.reshape(3, 1), np.zeros((3, 1))))
    
    # Solve using PG3 (assuming it returns a 2-element array of angles)
    thv = pardos_gotor_three(x3, r3, o1, nvp)
    thu = pardos_gotor_three(x3, r3, o2, nup)
    
    # Compute intersection points on the line
    pc1 = r3 + thv[0] * v3
    pd1 = r3 + thv[1] * v3
    pc2 = r3 + thu[0] * v3
    pd2 = r3 + thu[1] * v3
    
    # Apply PK1 subproblem approach to find angles
    m1 = pc1 - r1
    mp1 = m1 - w1 * (w1 @ m1)
    n1 = pd1 - r1
    np1 = n1 - w1 * (w1 @ n1)
    
    m2 = pc2 - r2
    mp2 = m2 - w2 * (w2 @ m2)
    n2 = pd2 - r2
    np2 = n2 - w2 * (w2 @ n2)
    
    t1c = np.arctan2(np.real(w1 @ np.cross(mp1, vp1)), np.real(mp1 @ vp1))
    t1d = np.arctan2(np.real(w1 @ np.cross(np1, vp1)), np.real(np1 @ vp1))
    t2c = np.arctan2(np.real(w2 @ np.cross(up2, mp2)), np.real(up2 @ mp2))
    t2d = np.arctan2(np.real(w2 @ np.cross(up2, np2)), np.real(up2 @ np2))
    
    return np.array([[t1c, t2c], [t1d, t2d]])

def pardos_gotor_seven(x1, x2, x3, pp, pk):
    """
    Solve the inverse kinematics problem for 3 consecutive rotations:
    1 skew (non-parallel) axis and 2 parallel axes using Pardos-Gotor Subproblem 7.

    Parameters:
        x1, x2, x3 : np.ndarray
            6x1 twists (v; w) for each screw motion.
        pp, pk : np.ndarray
            3x1 initial and final positions in SE(3).

    Returns:
        The123 : np.ndarray
            Array with up to 4 solutions, each as [theta1, theta2, theta3].
    """

    v1, w1 = x1[:3], x1[3:]
    v3, w3 = x3[:3], x3[3:]

    r1 = np.cross(w1, v1) / np.linalg.norm(w1)**2
    r3 = np.cross(w3, v3) / np.linalg.norm(w3)**2

    v = pk - r1
    vw1 = np.outer(w1, w1) @ v
    vp1 = v - vw1
    nvp = np.linalg.norm(vp1)
    o1 = r1 + vw1

    u = pp - r3
    uw3 = np.outer(w3, w3) @ u
    up3 = u - uw3
    nup = np.linalg.norm(up3)
    o3 = r3 + uw3

    # Planes in normal form
    d1 = w1 @ o1
    d3 = w3 @ o3

    # Direction of intersection line
    v4 = np.cross(w1, w3)
    x4 = np.concatenate([v4, np.zeros(3)]).reshape(-1, 1)  # Twist for PG3
    w13 = w1 @ w3

    # A point on the intersection line
    r4 = (w1 * (d1 - d3 * w13) + w3 * (d3 - d1 * w13)) / (1 - w13)

    # Find intersection points of the line with the circle using PG3
    t4 = pardos_gotor_three(x4, r4, o1, nvp)
    pc = r4 + t4[0] * v4
    pd = r4 + t4[1] * v4

    # Use PK1 to find theta1 (two solutions)
    m1 = pc - r1
    mp1 = m1 - (np.outer(w1, w1) @ m1)
    n1 = pd - r1
    np1 = n1 - (np.outer(w1, w1) @ n1)

    t1ck = np.arctan2(np.real(w1 @ np.cross(mp1, vp1)), np.real(mp1 @ vp1))
    t1dk = np.arctan2(np.real(w1 @ np.cross(np1, vp1)), np.real(np1 @ vp1))

    # Use PG4 to solve for theta2 and theta3 (each with 2 solutions)
    t23c = pardos_gotor_four(x2, x3, pp, pc)  # shape (2, 2)
    t23d = pardos_gotor_four(x2, x3, pp, pd)

    # Construct the final solution matrix
    The123 = np.array([
        [t1ck, t23c[0, 0], t23c[0, 1]],
        [t1ck, t23c[1, 0], t23c[1, 1]],
        [t1dk, t23d[0, 0], t23d[0, 1]],
        [t1dk, t23d[1, 0], t23d[1, 1]]
    ])

    return The123

def pardos_gotor_eight(x1, x2, x3, Hp, Hk):
    """
    PardosGotorEight:
    Solve the inverse kinematics for three consecutive parallel revolute joints using screw theory.
    
    Parameters:
        x1, x2, x3 : ndarray (6,)
            Twist vectors [v; w] for the three screw axes (assumed to be parallel).
        Hp : ndarray (4,4)
            Initial pose as a homogeneous transformation matrix.
        Hk : ndarray (4,4)
            Final pose as a homogeneous transformation matrix.

    Returns:
        The123 : ndarray (2, 3)
            Two possible sets of joint angles [theta1, theta2, theta3] that move Hp to Hk.
    """
    # Descomposición del tercer tornillo
    v3 = x3[0:3]
    w3 = x3[3:6]

    # Cálculo del punto r3 sobre el eje de rotación
    r3 = np.cross(w3, v3) / (np.linalg.norm(w3) ** 2)

    # Punto inicial en el espacio
    pp = Hp[0:3, 3]
    u = pp - r3
    o3p = r3 + np.outer(w3, w3) @ u  # Proyección ortogonal sobre el eje w3

    # Punto final deseado
    pk = Hk[0:3, 3]

    # Transformación del punto o3p mediante la pose relativa Hk * inv(Hp)
    o3ph = np.append(o3p, 1)
    o3kh = Hk @ np.linalg.inv(Hp) @ o3ph
    o3k = o3kh[0:3]

    # Resolver el subproblema PG4 con x1 y x2 para mover o3p hasta o3k
    t12 = pardos_gotor_four(x1, x2, o3p, o3k)  # Output: (2,2) → [[t1co, t2oc], [t1do, t2od]]

    # Obtener el vector twist del eje x3 pasando por o3k
    x3k = joint2twist(w3, o3k, 'rot')

    # Nuevo punto a transformar por el total de las tres rotaciones
    ppk = o3k + (pp - o3p)

    # Resolver el ángulo total de las tres rotaciones
    t123 = paden_kahan_one(x3k, ppk, pk)

    # Determinar t3 para cada solución parcial t1 y t2
    t31 = t123 - t12[0, 0] - t12[0, 1]
    t32 = t123 - t12[1, 0] - t12[1, 1]

    # Componer soluciones completas
    The123 = np.array([
        [t12[0, 0], t12[0, 1], t31],
        [t12[1, 0], t12[1, 1], t32]
    ])

    return The123

class NoParalelVectors(Exception):
    pass