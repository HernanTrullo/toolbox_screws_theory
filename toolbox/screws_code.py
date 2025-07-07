# Copyright (C) 2025 Hernan Trullo
#
# This file is part of [screws theory toolbox].
#
# [screws ik] is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# [screws ik] is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np

def intersect_lines_3d(axes, points):
    """
    Compute the intersection point of two 3D lines defined by direction vectors and points on each line.

    Parameters:
    - axes: (3, 2) ndarray, direction vectors of the two lines (columns are x1 and x2)
    - points: (3, 2) ndarray, points on the two lines (columns are p1 and p2)

    Returns:
    - m: (3,) ndarray, intersection point or [inf, inf, inf] if lines are parallel
    """
    c = points[:, 0]
    e = axes[:, 0]
    d = points[:, 1]
    f = axes[:, 1]
    
    g = d - c
    cfg = np.cross(f, g)
    cfe = np.cross(f, e)
    
    norm_cfe = np.linalg.norm(cfe)
    
    if norm_cfe == 0:
        return np.array([np.inf, np.inf, np.inf])
    elif np.dot(cfg, cfe) >= 0:
        return c + (np.linalg.norm(cfg) / norm_cfe) * e
    else:
        return c - (np.linalg.norm(cfg) / norm_cfe) * e
    

def axis2skew(w):
    """
    Generate a skew symmetric matrix from a 3x1 vector w.
    Useful for converting a cross product to a matrix multiplication:
    cross = np.cross(a, b) is equivalent to axis2skew(a) @ b

    Parameters:
    w (array-like): A 3-element vector [a1, a2, a3]

    Returns:
    numpy.ndarray: A 3x3 skew symmetric matrix
    """
    w = np.asarray(w)
    if w.shape != (3,):
        raise ValueError("Input vector must be a 3-element array")

    return np.array([
        [0,     -w[2],  w[1]],
        [w[2],   0,    -w[0]],
        [-w[1],  w[0],  0]
    ])
    
    
def expAxAng(AxAng):
    """
    Compute the matrix exponential of an axis-angle representation.

    Parameters:
    AxAng (array-like): A 4-element vector [x, y, z, theta]
        where [x, y, z] is the axis of rotation (must be normalized)
        and theta is the rotation angle in radians.

    Returns:
    numpy.ndarray: A 3x3 rotation matrix
    """
    AxAng = np.asarray(AxAng)
    if AxAng.shape != (4,):
        raise ValueError("Input must be a 4-element vector [x, y, z, theta]")

    w = AxAng[:3]
    theta = AxAng[3]
    ws = axis2skew(w)

    I = np.eye(3)
    rotm = I + np.sin(theta) * ws + (1 - np.cos(theta)) * (ws @ ws)
    return rotm


def expScrew(TwMag):
    """
    EXPSCREW Matrix Exponential of a Rigid Body Motion by SCREW movement.
    
    Parameters:
        TwMag: numpy array (7x1) -> [v (3,), w (3,), theta]
        
    Returns:
        H: Homogeneous transformation matrix (4x4)
    """
    v = TwMag[0:3]      # "vee" component of the TWIST
    w = TwMag[3:6]      # "omega" component of the TWIST
    t = TwMag[6]        # "theta" magnitude component of the SCREW

    if np.linalg.norm(w) == 0:  # Pure translation
        r = np.eye(3)
        p = v * t
    else:
        r = expAxAng(np.concatenate((w, [t])))  # Assume this returns a 3x3 rotation matrix
        p = (np.eye(3) - r) @ np.cross(w, v)    # For only rotation joint
        # For general screw motion (rotation + translation), uncomment this line:
        # p = (np.eye(3) - r) @ np.cross(w, v) + np.outer(w, w) @ v * t
        
    H = np.eye(4)
    H[0:3, 0:3] = r
    H[0:3, 3] = p
    return H


def joint2twist(Axis, Point, JointType):
    """
    Gets the TWIST from the Joint AXIS and a POINT on that axis.

    Parameters:
        Axis (array-like): 3-element array, the direction of the joint axis.
        Point (array-like): 3-element array, a point on the axis.
        JointType (str): 'rot' for rotational joint or 'tra' for translational joint.

    Returns:
        xi (numpy.ndarray): 6x1 twist vector.
    """
    Axis = np.asarray(Axis).flatten()
    Point = np.asarray(Point).flatten()

    if JointType == "rot":
        v = -np.cross(Axis, Point)
        w = Axis
        xi = np.concatenate((v, w))
    elif JointType == "tra":
        v = Axis
        w = np.array([0.0, 0.0, 0.0])
        xi = np.concatenate((v, w))
    else:
        xi = np.zeros(6)

    return xi.reshape(-1,1)

def forward_kinematics_poe(TwMag):
    """
    Forward Kinematics using the Product of Exponentials (POE) formula.

    Parameters:
        TwMag: np.ndarray of shape (7, n)
            Each column is a twist-magnitude vector:
            - First 6 elements are the twist (v, w)
            - Last element is the magnitude (theta)

    Returns:
        HstR: np.ndarray of shape (4, 4)
            Homogeneous transformation matrix representing the pose of
            the end-effector with respect to the base frame.
    """
    try:
        n = TwMag.shape[1]
    except:
        n =1
    HstR = expScrew(TwMag[:, 0])  # Initial transformation

    for i in range(1, n):
        HstR = HstR @ expScrew(TwMag[:, i])  # Matrix multiplication

    return HstR

"""# Número de grados de libertad
n = 6
# Generar magnitudes aleatorias theta1-theta6
Mag = np.ones(6)*np.pi/4

# Características mecánicas del robot IRB120
po = np.array([0, 0, 0])
pk = np.array([0, 0, 0.290])
pr = np.array([0, 0, 0.560])
pf = np.array([0.302, 0, 0.630])
pp = np.array([0.302, 0, 0.470])

AxisX = np.array([1, 0, 0])
AxisY = np.array([0, 1, 0])
AxisZ = np.array([0, 0, 1])

# Puntos en las articulaciones
Point = np.column_stack((po, pk, pr, pf, pf, pf))

# Tipos de articulaciones
Joint = ['rot', 'rot', 'rot', 'rot', 'rot', 'rot']

# Ejes de las articulaciones
Axis = np.column_stack((AxisZ, AxisY, AxisY, AxisX, AxisY, -AxisZ))

# Calcular twists
Twist = np.zeros((6, n))
for i in range(n):
    Twist[:, i] = joint2twist(Axis[:, i], Point[:, i], Joint[i])
    

print(Twist)
#Aplicar Forward Kinematics POE
TwMag = np.vstack((Twist, Mag))  # Apila Twist (6xn) y Mag (1xn)
print(TwMag)
HstR = forward_kinematics_poe(TwMag)

# matriz homogénea del efector final
print("HstR (matriz homogénea final del efector):\n", HstR)"""







