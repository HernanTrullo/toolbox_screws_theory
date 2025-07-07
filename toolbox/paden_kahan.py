import numpy as np
from toolbox.screws_code import intersect_lines_3d

def paden_kahan_one(x1, pp, pk):
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
    theta1 = np.arctan2(np.dot(w1, np.cross(up, vp)), np.dot(up, vp))
    
    return theta1

def paden_kahan_two(x1, x2, pp, pk):
    """
    Solve Paden-Kahan subproblem 2 for two consecutive rotation screws.

    Parameters:
    - x1: (6,) twist vector (v1; w1)
    - x2: (6,) twist vector (v2; w2)
    - pp: (3,) initial point
    - pk: (3,) final point

    Returns:
    - Theta1Theta2: (2, 2) ndarray with solutions [[t11, t21], [t12, t22]]
    """
    v1, w1 = x1[:3], x1[3:]
    v2, w2 = x2[:3], x2[3:]
    r1 = np.cross(w1, v1) / np.linalg.norm(w1)**2
    r2 = np.cross(w2, v2) / np.linalg.norm(w2)**2

    pr = intersect_lines_3d(np.column_stack((w1, w2)), np.column_stack((r1, r2)))

    if np.any(np.isinf(pr)):
        return np.array([[0, 0], [0, 0]])

    u = pp - pr
    v = pk - pr
    Cw1w2 = np.cross(w1, w2)

    dot_w1w2 = np.dot(w1, w2)
    denom = dot_w1w2**2 - 1

    a = (dot_w1w2 * np.dot(w2, u) - np.dot(w1, v)) / denom
    b = (dot_w1w2 * np.dot(w1, v) - np.dot(w2, u)) / denom

    g2_num = np.linalg.norm(u)**2 - a**2 - b**2 - 2 * a * b * dot_w1w2
    g2 = abs(np.real(g2_num) / np.linalg.norm(Cw1w2)**2)
    g = np.sqrt(g2)

    pc = pr + a * w1 + b * w2 + g * Cw1w2
    pd = pr + a * w1 + b * w2 - g * Cw1w2

    m = pc - pr
    n = pd - pr

    # Solve two double solutions
    proj = lambda x, w: x - np.dot(w, x) * w

    up = proj(u, w2)
    m2p = proj(m, w2)
    m1p = proj(m, w1)
    n2p = proj(n, w2)
    n1p = proj(n, w1)
    vp = proj(v, w1)

    t21 = np.arctan2(np.real(np.dot(w2, np.cross(up, m2p))), np.real(np.dot(up, m2p)))
    t11 = np.arctan2(np.real(np.dot(w1, np.cross(m1p, vp))), np.real(np.dot(m1p, vp)))
    t22 = np.arctan2(np.real(np.dot(w2, np.cross(up, n2p))), np.real(np.dot(up, n2p)))
    t12 = np.arctan2(np.real(np.dot(w1, np.cross(n1p, vp))), np.real(np.dot(n1p, vp)))

    return np.array([[t11, t21], [t12, t22]])


def paden_kahan_three(x1, pp, pk, de):
    """
    Solves the Paden-Kahan Subproblem 3 (PK3) in screw theory for inverse kinematics.

    Given a screw axis (x1), two points (pp and pk), and a scalar distance (de), this function
    finds the possible rotation angles (theta1) about the axis such that the rotated point `pp`
    lies on a sphere of radius `de` centered at `pk`. This subproblem arises when the intersection
    between a helical trajectory and a sphere is sought, typically in the context of inverse kinematics.

    Parameters:
    ----------
    x1 : ndarray of shape (6,1)
        The screw axis represented as a 6D vector [v; w], where:
        - v is the linear velocity vector (first 3 elements)
        - w is the angular velocity vector (last 3 elements), assumed to be non-zero
    pp : ndarray of shape (3,)
        The initial point to be rotated.
    pk : ndarray of shape (3,)
        The center of the sphere.
    de : float
        The desired distance (radius of the sphere) between the rotated point and `pk`.

    Returns:
    -------
    theta1 : ndarray of shape (2,)
        The two possible solutions (in radians) for the rotation angle theta1
        that satisfy the geometric constraints of the problem.
    """
    v1 = x1[0:3,0]
    w1 = x1[3:,0]
    w1 = w1 / np.linalg.norm(w1)  # Asegura que w1 es unitario
    r1 = np.cross(w1, v1) / np.linalg.norm(w1)**2

    u = pp - r1
    up = u - np.dot(w1, u) * w1
    nup = np.linalg.norm(up)

    v = pk - r1
    vp = v - np.dot(w1, v) * w1
    nvp = np.linalg.norm(vp)

    alfa1 = np.arctan2(np.real(np.dot(w1, np.cross(up, vp))), np.real(np.dot(up, vp)))

    dep2 = np.real(de**2 - np.linalg.norm(np.dot(w1, pp - pk))**2)
    beta = (nup**2 + nvp**2 - dep2) / (2 * nup * nvp)

    # Limitar beta a [-1, 1] por precisión numérica
    beta = np.clip(beta, -1.0, 1.0)

    beta1 = np.arccos(beta)

    theta1 = np.array([alfa1 - beta1, alfa1 + beta1])
    return theta1

