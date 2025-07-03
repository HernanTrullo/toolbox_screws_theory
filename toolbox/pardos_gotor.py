import numpy as np
from toolbox.screws_code import intersect_lines_3d

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

class NoParalelVectors(Exception):
    pass