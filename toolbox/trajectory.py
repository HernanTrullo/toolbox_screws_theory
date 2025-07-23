import numpy as np

def genrate_trajectory_quintic(Tem, Qini, Qfin, Tfin):
    """
    Generates a smooth trajectory using a quintic profile (5th-order polynomial).

    Parameters:
    Tem -- sampling time (seconds)
    Qini -- initial position 
    Qfin -- final position
    Tfin -- total trajectory time (seconds)

    Returns:
    t -- time vector
    qd_1 -- generated trajectory
    """
    t = np.arange(0, Tfin + Tem, Tem)  # Vector de tiempo
    delta_q = Qfin - Qini              # Distancia a recorrer

    # Perfil quintic normalizado
    s = (10 * (t / Tfin) ** 3
         - 15 * (t / Tfin) ** 4
         + 6 * (t / Tfin) ** 5)

    # Trayectoria completa
    qd_1 = Qini + delta_q * s

    return qd_1

def normalize_angle_rad(angle):
    """
    Normaliza un ángulo en radianes al rango [-π, π).
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi
