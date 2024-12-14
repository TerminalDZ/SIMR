import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.special import gamma
import plotly.express as px

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

# Création des visualisations
st.header("Visualisation des Résultats")

# Options d'affichage 3D
st.sidebar.header("Options d'affichage 3D")
colormap = st.sidebar.selectbox(
    "Choisir la palette de couleurs",
    ["viridis", "plasma", "inferno", "magma", "hot", "cool"]
)

# Création des graphiques 3D
col1, col2 = st.columns(2)

with col1:
    st.subheader("Déplacement u(x, t) en 3D")
    X, T = np.meshgrid(x, t)
    
    fig_u = go.Figure(data=[go.Surface(
        x=X, y=T, z=u,
        colorscale=colormap,
        showscale=True,
        colorbar=dict(title="Déplacement (m)")
    )])
    
    fig_u.update_layout(
        scene=dict(
            xaxis_title="Position x (m)",
            yaxis_title="Temps t (s)",
            zaxis_title="Déplacement u (m)",
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=700,
        height=600,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    st.plotly_chart(fig_u, use_container_width=True)

with col2:
    st.subheader("Potentiel électrique φ(x, t) en 3D")
    
    fig_phi = go.Figure(data=[go.Surface(
        x=X, y=T, z=phi,
        colorscale=colormap,
        showscale=True,
        colorbar=dict(title="Potentiel (V)")
    )])
    
    fig_phi.update_layout(
        scene=dict(
            xaxis_title="Position x (m)",
            yaxis_title="Temps t (s)",
            zaxis_title="Potentiel φ (V)",
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=700,
        height=600,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    st.plotly_chart(fig_phi, use_container_width=True)

# Instructions pour l'interaction
st.sidebar.markdown("""
### Instructions pour l'interaction 3D:
- **Rotation**: Cliquez et faites glisser
- **Zoom**: Utilisez la molette de la souris ou pincez sur l'écran tactile
- **Déplacement**: Cliquez-droit et faites glisser
- **Reset**: Double-cliquez sur le graphique
- **Vue spécifique**: Utilisez les boutons de la barre d'outils
""")

# Affichage des valeurs à un temps spécifique
st.header("Visualisation 2D à un temps spécifique")
selected_time = st.slider("Sélectionner le temps (s)", min_value=0.0, max_value=float(T), value=float(T)/2)
time_index = int((selected_time/T) * (N_t-1))

col3, col4 = st.columns(2)

with col3:
    fig_u_2d = px.line(
        x=x, y=u[time_index, :],
        labels={"x": "Position x (m)", "y": "Déplacement u (m)"},
        title=f"Déplacement à t = {selected_time:.2f} s"
    )
    fig_u_2d.update_traces(line_color='blue')
    st.plotly_chart(fig_u_2d, use_container_width=True)

with col4:
    fig_phi_2d = px.line(
        x=x, y=phi[time_index, :],
        labels={"x": "Position x (m)", "y": "Potentiel électrique φ (V)"},
        title=f"Potentiel électrique à t = {selected_time:.2f} s"
    )
    fig_phi_2d.update_traces(line_color='red')
    st.plotly_chart(fig_phi_2d, use_container_width=True)