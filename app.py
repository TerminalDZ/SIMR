import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from matplotlib import cm

st.set_page_config(layout="wide", page_title="Simulation de Barre Piézoélectrique")

st.title("Simulation de Barre Piézoélectrique")

# Sidebar pour les paramètres
st.sidebar.header("Paramètres de Simulation")

# Paramètres du problème
col1, col2 = st.sidebar.columns(2)
with col1:
    L = st.number_input("Longueur de la barre (m)", value=1.0, min_value=0.1)
    N_x = st.number_input("Nombre de points spatiaux", value=100, min_value=10)
    A = st.number_input("Constante A", value=1.0, min_value=0.1)
    E = st.number_input("Module d'élasticité", value=1.0, min_value=0.1)
    C0 = st.number_input("Constante C0", value=1.0, min_value=0.1)

with col2:
    T = st.number_input("Temps final (s)", value=1.0, min_value=0.1)
    N_t = st.number_input("Nombre de points temporels", value=100, min_value=10)
    B = st.number_input("Constante B", value=1.0, min_value=0.1)
    E_star = st.number_input("Constante E*", value=1.0, min_value=0.1)
    alpha = st.slider("Ordre fractionnaire", min_value=0.1, max_value=1.0, value=0.5, step=0.1)

u1_0 = st.sidebar.number_input("Valeur initiale de u1", value=1.0)

# Calcul des constantes
k1 = B + (E * E_star) / B
k2 = A * C0
lambd = k1 / k2

# Grilles spatiales et temporelles
x = np.linspace(0, L, N_x)
t = np.linspace(0, T, N_t)

# Fonction d'approximation de Mittag-Leffler
def mittag_leffler_approx(alpha, z, n_terms=50):
    sum_ml = 0.0
    for n in range(n_terms):
        sum_ml += z**n / gamma(alpha * n + 1)
    return sum_ml

# Calcul des solutions
@st.cache_data
def calculate_solutions(x, t, u1_0, alpha, lambd):
    N_t, N_x = len(t), len(x)
    u = np.zeros((N_t, N_x))
    phi = np.zeros((N_t, N_x))
    
    for i in range(N_t):
        ti = t[i]
        E_alpha = mittag_leffler_approx(alpha, -lambd * ti**alpha)
        u1_t = u1_0 * E_alpha
        u_t = u1_t * x
        phi_t = (E / B) * u_t
        u[i, :] = u_t
        phi[i, :] = phi_t
    
    return u, phi

# Calcul des solutions
u, phi = calculate_solutions(x, t, u1_0, alpha, lambd)

# Affichage des résultats
st.header("Visualisation des Résultats")

# Création de deux colonnes pour les graphiques 2D
col1, col2 = st.columns(2)

with col1:
    st.subheader("Déplacement u(x, t) à différents instants")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for i in range(0, N_t, N_t // 5):
        ax1.plot(x, u[i, :], label=f"t = {t[i]:.2f} s")
    ax1.set_xlabel("Position x (m)")
    ax1.set_ylabel("Déplacement u (m)")
    ax1.grid(True)
    ax1.legend()
    st.pyplot(fig1)

with col2:
    st.subheader("Potentiel électrique φ(x, t) à différents instants")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for i in range(0, N_t, N_t // 5):
        ax2.plot(x, phi[i, :], label=f"t = {t[i]:.2f} s")
    ax2.set_xlabel("Position x (m)")
    ax2.set_ylabel("Potentiel électrique φ (V)")
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)

# Création de deux colonnes pour les graphiques 3D
col3, col4 = st.columns(2)

with col3:
    st.subheader("Déplacement u(x, t) en 3D")
    X, T = np.meshgrid(x, t)
    fig3 = plt.figure(figsize=(10, 6))
    ax3 = plt.axes(projection='3d')
    surf1 = ax3.plot_surface(X, T, u, cmap=cm.viridis)
    ax3.set_xlabel("Position x (m)")
    ax3.set_ylabel("Temps t (s)")
    ax3.set_zlabel("Déplacement u (m)")
    fig3.colorbar(surf1)
    st.pyplot(fig3)

with col4:
    st.subheader("Potentiel électrique φ(x, t) en 3D")
    fig4 = plt.figure(figsize=(10, 6))
    ax4 = plt.axes(projection='3d')
    surf2 = ax4.plot_surface(X, T, phi, cmap=cm.plasma)
    ax4.set_xlabel("Position x (m)")
    ax4.set_ylabel("Temps t (s)")
    ax4.set_zlabel("Potentiel électrique φ (V)")
    fig4.colorbar(surf2)
    st.pyplot(fig4)