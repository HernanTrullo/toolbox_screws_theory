import numpy as np

# Ángulo de rotación en radianes (por ejemplo, 45°)
theta = np.deg2rad(45)

# Matriz de rotación alrededor del eje Z
R = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta),  np.cos(theta), 0],
    [0,              0,             1]
])

# Transpuesta de R
Rt = R.T

# Producto R · Rᵀ
identity_approx = R @ Rt

# Mostrar resultados
print("R:")
print(R)

print("\nR.T (transpuesta):")
print(Rt)

print("\nR @ R.T:")
print(identity_approx)

print("\n¿Es identidad? (usando np.allclose):", np.allclose(identity_approx, np.eye(3)))
