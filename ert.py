import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from mpl_toolkits.mplot3d import Axes3D

# Paramètres du problème
Lx, Ly, Lz = 1.0, 1.0, 1.0   # Dimensions du domaine
T = 1.0                      # Temps final
Nx, Ny, Nz = 50, 50, 50      # Nombre de points spatiaux
Nt = 100                     # Nombre de points temporels

# Constantes matérielles
A = 1.0
B = 1.0
E = 1.0
E_star = 1.0
C0 = 1.0
alpha = 0.5                  # Ordre fractionnaire (0 < alpha <= 1)
k1 = B + (E * E_star) / B
k2 = A * C0
lambd = k1 / k2
u1_0 = 1.0                   # Valeur initiale de u1

# Grilles spatiales et temporelles
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
z = np.linspace(0, Lz, Nz)
t = np.linspace(0, T, Nt)
X2D, Y2D = np.meshgrid(x, y, indexing='ij')
X3D, Y3D, Z3D = np.meshgrid(x, y, z, indexing='ij')

# Fonction de Mittag-Leffler
def mittag_leffler(alpha, z, n_terms=100):
    ml = np.zeros_like(z, dtype=np.float64)
    for n in range(n_terms):
        ml += z**n / gamma(alpha * n + 1)
    return ml

# Préparation des matrices de solution
U2D = np.zeros((Nx, Ny, Nt))
Phi2D = np.zeros((Nx, Ny, Nt))
U3D = np.zeros((Nx, Ny, Nz, Nt))
Phi3D = np.zeros((Nx, Ny, Nz, Nt))

# Calcul de la solution à chaque instant
for k in range(Nt):
    tk = t[k]
    E_alpha = mittag_leffler(alpha, -lambd * tk**alpha)
    u1_t = u1_0 * E_alpha

    # Simulation 2D
    U2D[:, :, k] = u1_t * (X2D + Y2D)
    Phi2D[:, :, k] = (E / B) * U2D[:, :, k]

    # Simulation 3D
    U3D[:, :, :, k] = u1_t * (X3D + Y3D + Z3D)
    Phi3D[:, :, :, k] = (E / B) * U3D[:, :, :, k]

# Visualisation des résultats en 2D à t = T
plt.figure(figsize=(8, 6))
plt.contourf(X2D, Y2D, U2D[:, :, -1], cmap='viridis')
plt.title("Déplacement u(x, y, t=T) en 2D")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar(label="Déplacement u")
plt.show()

plt.figure(figsize=(8, 6))
plt.contourf(X2D, Y2D, Phi2D[:, :, -1], cmap='plasma')
plt.title("Potentiel électrique φ(x, y, t=T) en 2D")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar(label="Potentiel φ")
plt.show()

# Visualisation des résultats en 3D à t = T
# Déplacement u(x, y, z=Lz/2, t=T)
z_index = Nz // 2

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection='3d')

step = 4  # Pour une meilleure performance de tracé
X_plot = X3D[:, :, z_index]
Y_plot = Y3D[:, :, z_index]
U_plot = U3D[:, :, z_index, -1]

ax.plot_surface(X_plot[::step, ::step], Y_plot[::step, ::step], U_plot[::step, ::step], cmap='viridis')
ax.set_title("Déplacement u(x, y, z=Lz/2, t=T) en 3D")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Déplacement u")

# Potentiel électrique φ(x, y, z=Lz/2, t=T)
ax2 = fig.add_subplot(122, projection='3d')
Phi_plot = Phi3D[:, :, z_index, -1]
ax2.plot_surface(X_plot[::step, ::step], Y_plot[::step, ::step], Phi_plot[::step, ::step], cmap='plasma')
ax2.set_title("Potentiel électrique φ(x, y, z=Lz/2, t=T) en 3D")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("Potentiel φ")

plt.tight_layout()
plt.show()