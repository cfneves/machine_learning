"""
=============================================================================
Machine Learning — Aula 07 / K-Means, PCA e Redução de Dimensionalidade
Aplicação Streamlit interativa — material didático para iniciantes

Autor       : Cláudio Ferreira Neves
Cargo atual : Analista de BI — Save Co. / Especialista de Ensino II — SENAI/SC
Certificação: DATA ANALYST CERTIFIED PROFESSIONAL © (DACP)
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    homogeneity_score, completeness_score, v_measure_score
)
from sklearn.datasets import load_breast_cancer

PAGE_PORTAL  = "pages/Portal.py"
PAGE_AULA_06 = "pages/Aula_06.py"
PAGE_AULA_08 = "pages/Aula_08.py"

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────────────
def save_and_show(fig, filename):
    fig.savefig(os.path.join(OUTPUTS_DIR, f"{filename}.png"), dpi=150, bbox_inches="tight")
    st.pyplot(fig); plt.close(fig)

def caixa(html, cor="#f0f4ff", borda="#667eea"):
    st.markdown(f'<div style="background:{cor};border-left:5px solid {borda};'
                f'padding:1rem 1.5rem;border-radius:0 10px 10px 0;margin:.8rem 0;">'
                f'{html}</div>', unsafe_allow_html=True)

def divider(): st.markdown("---")
def secao(t):  st.markdown(f"### {t}")
def code_block(code, title=""):
    if title: st.markdown(f"**{title}**")
    st.code(code, language="python")

# ── dados ─────────────────────────────────────────────────────────────────────
MALL_URL = ("https://raw.githubusercontent.com/matheusvanzan/"
            "Machine-Learning-Examples/refs/heads/master/datasets/Mall_Customers.csv")

@st.cache_data
def carregar_mall():
    return pd.read_csv(MALL_URL).dropna()

@st.cache_data
def carregar_breast_cancer_cluster():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target_names[data.target]
    return df

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="K-Means e PCA", page_icon="🔵",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
html,body,[class*="css"]{font-family:'Segoe UI',sans-serif;}
.metric-card{background:linear-gradient(135deg,#667eea,#764ba2);border-radius:12px;
  padding:1rem 1.5rem;color:white;text-align:center;margin-bottom:.5rem;
  box-shadow:0 4px 15px rgba(102,126,234,.4);}
.metric-card h2{margin:0;font-size:2rem;} .metric-card p{margin:0;opacity:.85;font-size:.9rem;}
.sidebar-header{background:linear-gradient(135deg,#1a1a2e,#16213e);padding:1.2rem;
  border-radius:10px;text-align:center;margin-bottom:1rem;color:white;}
.footer{background:linear-gradient(135deg,#1a1a2e,#16213e);color:white;
  padding:2rem;border-radius:12px;text-align:center;margin-top:3rem;}
</style>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-header"><h2 style="margin:0;font-size:1.3rem;">🔵 Aula 07</h2>'
                '<p style="margin:.4rem 0 0;font-size:.8rem;opacity:.8;">K-Means e PCA</p></div>',
                unsafe_allow_html=True)
    pagina = st.radio("Nav", [
        "🏠  Início",
        "🔵  K-Means — Teoria",
        "🛍️  K-Means — Mall Customers",
        "📐  Elbow e Silhouette",
        "🌀  PCA — Redução de Dimensionalidade",
        "💊  Exercício — Breast Cancer",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("<div style='font-size:.75rem;color:#888;text-align:center;'>"
                "📁 Gráficos em <code>outputs/</code></div>", unsafe_allow_html=True)

# ── navegação entre aulas ────────────────────────────────────────────────────
from nav import tab_nav
tab_nav(7)  # replace CURRENT_NUMBER with the correct int

st.markdown("<p style='text-align:center;font-size:.95rem;color:#667eea;font-weight:600;margin-bottom:0;'>"
            "Autor: Especialista Cláudio Ferreira Neves</p>", unsafe_allow_html=True)
nl, nm, nr = st.columns([1.2, 4, 1.2])
with nl:
    if st.button("← Aula 06", use_container_width=True, key="nav_prev"):
        st.switch_page(PAGE_AULA_06)
with nm:
    if st.button("🏠 Portal", use_container_width=True, key="nav_portal"):
        st.switch_page(PAGE_PORTAL)
with nr:
    if st.button("Aula 08 →", use_container_width=True, key="nav_next"):
        st.switch_page(PAGE_AULA_08)

# ============================================================================
# INÍCIO
# ============================================================================
if pagina == "🏠  Início":
    st.markdown("<div style='text-align:center;padding:2rem 0 1rem;'>"
                "<h1 style='font-size:2.8rem;font-weight:800;color:#1a1a2e;'>🔵 K-Means e PCA</h1>"
                "<p style='font-size:1.15rem;color:#555;max-width:720px;margin:0 auto;'>"
                "Aprendizado <b>não supervisionado</b>: descubra padrões e grupos nos dados "
                "sem usar rótulos — e visualize dados de alta dimensão em 2D.</p></div>",
                unsafe_allow_html=True)
    divider()
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown('<div class="metric-card"><h2>🔵</h2><p>K-Means</p></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="metric-card"><h2>🌀</h2><p>PCA</p></div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="metric-card"><h2>🛍️</h2><p>Mall Customers</p></div>', unsafe_allow_html=True)
    with c4: st.markdown('<div class="metric-card"><h2>📐</h2><p>Elbow + Silhouette</p></div>', unsafe_allow_html=True)
    divider()
    st.markdown("## 📚 O que você vai aprender")
    ca, cb = st.columns(2)
    with ca:
        caixa("<b>🔵 K-Means — Clusterização</b><br>"
              "Algoritmo <b>não supervisionado</b> que agrupa dados em K clusters "
              "minimizando a variância interna. Cada ponto é atribuído ao centroide "
              "mais próximo. Iterativo até a convergência.")
        caixa("<b>📐 Escolha do K — Elbow e Silhouette</b><br>"
              "• <b>Método do Cotovelo (Elbow)</b>: plota inércia vs K e identifica "
              "o ponto onde a queda desacelera<br>"
              "• <b>Silhouette Score</b>: mede coesão e separação dos clusters "
              "(-1 a 1, quanto maior melhor)")
        caixa("<b>🛍️ Dataset Mall Customers</b><br>"
              "200 clientes de shopping com renda anual e índice de gastos. "
              "Segmentação de clientes — caso de uso real de K-Means em marketing.")
    with cb:
        caixa("<b>🌀 PCA — Principal Component Analysis</b><br>"
              "Redução de dimensionalidade <b>linear</b>. Transforma os dados em "
              "componentes principais que explicam a maior variância. "
              "Útil para visualizar dados de alta dimensão em 2D.")
        caixa("<b>📊 Métricas de Avaliação</b><br>"
              "• <b>Inércia (WCSS)</b>: soma das distâncias ao centroide (menor = melhor)<br>"
              "• <b>Silhouette</b>: coesão e separação (-1 a 1)<br>"
              "• <b>CH Index</b>: razão variância entre/dentro clusters (maior = melhor)<br>"
              "• <b>DB Index</b>: semelhança entre clusters (menor = melhor)")
        caixa("<b>💊 Exercício — Breast Cancer + PCA</b><br>"
              "Aplique PCA para reduzir 30 features para 2 componentes e "
              "use K-Means para clusterizar. Compare os clusters com os "
              "rótulos reais — visualização clara da separabilidade.")
    divider()
    st.markdown("""<div class="footer">
        <p style="margin:0;font-size:1.1rem;font-weight:700;">Machine Learning — Aula 07</p>
        <p style="margin:.3rem 0 0;font-size:.9rem;opacity:.8;">
        K-Means e PCA · Cláudio Ferreira Neves · SENAI/SC</p></div>""",
        unsafe_allow_html=True)

# ============================================================================
# K-MEANS — TEORIA
# ============================================================================
elif pagina == "🔵  K-Means — Teoria":
    st.markdown('<h2>🔵 K-Means — Fundamentos</h2>', unsafe_allow_html=True)
    divider()

    caixa("<b>O que é K-Means?</b><br>"
          "Algoritmo <b>não supervisionado</b> para agrupamento de dados. "
          "Divide um conjunto em <b>K grupos</b> (clusters) onde os pontos de cada grupo "
          "são mais semelhantes entre si do que aos demais. "
          "Usa <b>distância Euclidiana</b> e busca minimizar a variância intra-cluster.")

    divider()
    secao("Algoritmo passo a passo")
    cp1, cp2 = st.columns(2)
    with cp1:
        caixa("<b>1. Inicialização</b><br>Escolha K centroides iniciais aleatórios "
              "(ou via K-Means++, que é mais inteligente).", cor="#f0f8ff", borda="#4C72B0")
        caixa("<b>2. Atribuição</b><br>Cada ponto é atribuído ao centroide mais próximo "
              "pela <b>distância Euclidiana</b>.",
              cor="#f0f8ff", borda="#4C72B0")
        caixa("<b>3. Atualização</b><br>Recalcula cada centroide como a <b>média</b> "
              "dos pontos do cluster.",
              cor="#f0f8ff", borda="#4C72B0")
    with cp2:
        caixa("<b>4. Repetição</b><br>Repete os passos 2 e 3 até que:<br>"
              "• Os centroides não mudem (convergência)<br>"
              "• Ou o número máximo de iterações seja atingido.", cor="#f0f8ff", borda="#4C72B0")
        caixa("<b>⚠️ K-Means precisa de escalonamento!</b><br>"
              "Usa distâncias Euclidianas → features em escalas diferentes "
              "têm pesos diferentes. Uma feature em milhares domina features "
              "em unidades. Sempre use <code>StandardScaler</code>.",
              cor="#fff0f0", borda="#e53e3e")
        caixa("<b>⚠️ K deve ser definido antes</b><br>"
              "Diferente de algoritmos supervisionados, o K-Means exige que você "
              "defina o número de clusters com antecedência. Use o método "
              "Elbow e Silhouette para estimar o melhor K.",
              cor="#fff8e1", borda="#f59e0b")

    divider()
    secao("Demonstração Interativa — K-Means 2D")

    np.random.seed(42)
    n_clusters_demo = st.slider("Número de clusters (K):", 2, 6, 3, 1, key="k_demo")

    cluster1 = np.random.randn(50,2) + np.array([2,2])
    cluster2 = np.random.randn(50,2) + np.array([-2,-2])
    cluster3 = np.random.randn(50,2) + np.array([2,-2])
    data_demo = np.vstack([cluster1, cluster2, cluster3])
    y_true_demo = np.array([0]*50 + [1]*50 + [2]*50)

    km_demo = KMeans(n_clusters=n_clusters_demo, random_state=42, n_init=10)
    km_demo.fit(data_demo)
    labels_demo = km_demo.labels_
    centroids_demo = km_demo.cluster_centers_
    inertia_demo   = km_demo.inertia_
    sil_demo = silhouette_score(data_demo, labels_demo) if n_clusters_demo > 1 else 0.0

    cdi1, cdi2, cdi3 = st.columns(3)
    cdi1.metric("Inércia", f"{inertia_demo:.2f}")
    cdi2.metric("Silhouette", f"{sil_demo:.4f}")
    cdi3.metric("K selecionado", str(n_clusters_demo))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cores_reais = ["#4C72B0","#DD8452","#55A868"]
    for i in range(3):
        axes[0].scatter(data_demo[y_true_demo==i,0], data_demo[y_true_demo==i,1],
                        color=cores_reais[i], s=30, alpha=0.7, label=f"Grupo {i+1}")
    axes[0].set_title("Dados (grupos reais)"); axes[0].legend(); axes[0].grid(alpha=0.3)

    cmap = plt.cm.get_cmap("tab10", n_clusters_demo)
    axes[1].scatter(data_demo[:,0], data_demo[:,1], c=labels_demo,
                    cmap=cmap, s=30, alpha=0.7)
    axes[1].scatter(centroids_demo[:,0], centroids_demo[:,1],
                    c="red", marker="X", s=200, zorder=5, label="Centroides")
    axes[1].set_title(f"K-Means (K={n_clusters_demo})")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    save_and_show(fig, "kmeans_demo_2d")

    divider()
    with st.expander("Ver código Python — K-Means básico"):
        code_block("""
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score

# Pipeline com escalonamento (OBRIGATÓRIO para K-Means)
pipe_kmeans = Pipeline([
    ('scl', StandardScaler()),
    ('kmeans', KMeans(n_clusters=5, n_init=10, random_state=42))
])
pipe_kmeans.fit(X)

# Rótulos e centroides
labels    = pipe_kmeans.named_steps['kmeans'].labels_
centroids = pipe_kmeans.named_steps['kmeans'].cluster_centers_

# Métricas
inertia   = pipe_kmeans.named_steps['kmeans'].inertia_
silhouette = silhouette_score(X, labels)
print(f"Inércia:    {inertia:.2f}")
print(f"Silhouette: {silhouette:.4f}")
""")

# ============================================================================
# K-MEANS — MALL CUSTOMERS
# ============================================================================
elif pagina == "🛍️  K-Means — Mall Customers":
    st.markdown('<h2>🛍️ K-Means — Segmentação de Clientes (Mall Customers)</h2>',
                unsafe_allow_html=True)
    divider()

    caixa("<b>Caso de uso real: Segmentação de Clientes</b><br>"
          "200 clientes de um shopping com <b>renda anual</b> e <b>índice de gastos</b>. "
          "Objetivo: identificar grupos com perfis semelhantes para ações de marketing "
          "personalizadas — sem usar nenhum rótulo manual.")

    with st.spinner("Carregando Mall Customers..."):
        try:
            df_mall = carregar_mall()
            st.success(f"Dataset: {df_mall.shape[0]} clientes × {df_mall.shape[1]} colunas")
        except Exception as e:
            st.error(f"Erro ao carregar: {e}")
            st.stop()

    secao("Análise Exploratória")
    cm1, cm2 = st.columns(2)
    with cm1:
        st.markdown("**Primeiras linhas:**")
        st.dataframe(df_mall.head(8), use_container_width=True)
    with cm2:
        st.markdown("**Estatísticas descritivas:**")
        st.dataframe(df_mall.describe().round(2), use_container_width=True)

    divider()
    secao("Scatter: Renda × Índice de Gastos")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df_mall["Annual_Income_(k$)"], df_mall["Spending_Score"],
               color="#667eea", alpha=0.7, s=40)
    ax.set_xlabel("Annual Income (k$)"); ax.set_ylabel("Spending Score")
    ax.set_title("Mall Customers — Renda vs Gastos")
    ax.grid(True, alpha=0.3)
    save_and_show(fig, "mall_scatter_raw")

    divider()
    secao("K-Means com 2 features (Renda + Gastos)")

    k_mall = st.slider("K (número de clusters):", 2, 10, 5, 1, key="k_mall")

    df_2d = df_mall[["Annual_Income_(k$)", "Spending_Score"]]
    numerical_mall = df_2d.columns.tolist()

    pre_mall = ColumnTransformer([("num", StandardScaler(), numerical_mall)])
    pipe_mall = Pipeline([("pre", pre_mall),
                          ("kmeans", KMeans(n_clusters=k_mall, n_init=10, random_state=42))])
    pipe_mall.fit(df_2d)

    labels_mall    = pipe_mall.predict(df_2d)
    centroids_scl  = pipe_mall.named_steps["kmeans"].cluster_centers_
    scaler_mall    = pipe_mall.named_steps["pre"].named_transformers_["num"]
    centroids_orig = scaler_mall.inverse_transform(centroids_scl)
    inertia_mall   = pipe_mall.named_steps["kmeans"].inertia_
    sil_mall = silhouette_score(df_2d, labels_mall) if k_mall > 1 else 0.0

    cmk1, cmk2, cmk3 = st.columns(3)
    cmk1.metric("Inércia", f"{inertia_mall:.2f}")
    cmk2.metric("Silhouette", f"{sil_mall:.4f}")
    cmk3.metric("K", str(k_mall))

    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.cm.get_cmap("tab10", k_mall)
    ax.scatter(df_2d["Annual_Income_(k$)"], df_2d["Spending_Score"],
               c=labels_mall, cmap=cmap, alpha=0.7, s=50)
    ax.scatter(centroids_orig[:,0], centroids_orig[:,1],
               c="red", marker="X", s=200, zorder=5, label="Centroides")
    ax.set_xlabel("Annual Income (k$)"); ax.set_ylabel("Spending Score")
    ax.set_title(f"K-Means (K={k_mall}) — Mall Customers")
    ax.legend(); ax.grid(alpha=0.3)
    save_and_show(fig, f"mall_kmeans_k{k_mall}")

    divider()
    with st.expander("Ver código Python — K-Means Mall Customers"):
        code_block("""
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df_2d = df[["Annual_Income_(k$)", "Spending_Score"]]

pre = ColumnTransformer([("num", StandardScaler(), df_2d.columns.tolist())])
pipe = Pipeline([
    ("pre",    pre),
    ("kmeans", KMeans(n_clusters=5, n_init=10, random_state=42))
])
pipe.fit(df_2d)

labels = pipe.predict(df_2d)
centroids_scl  = pipe.named_steps['kmeans'].cluster_centers_
scaler         = pipe.named_steps['pre'].named_transformers_['num']
centroids_orig = scaler.inverse_transform(centroids_scl)

import matplotlib.pyplot as plt
plt.scatter(df_2d.iloc[:,0], df_2d.iloc[:,1], c=labels, cmap='tab10', alpha=0.7)
plt.scatter(centroids_orig[:,0], centroids_orig[:,1],
            c='red', marker='X', s=200, label='Centroides')
plt.legend(); plt.show()
""")

# ============================================================================
# ELBOW E SILHOUETTE
# ============================================================================
elif pagina == "📐  Elbow e Silhouette":
    st.markdown('<h2>📐 Elbow e Silhouette — Encontrando o K ideal</h2>',
                unsafe_allow_html=True)
    divider()

    caixa("<b>Como escolher K?</b><br>"
          "O K-Means não escolhe K automaticamente — você precisa definir. "
          "Dois métodos clássicos para estimar o melhor K:<br><br>"
          "• <b>Método do Cotovelo (Elbow)</b>: plota a inércia para cada K. "
          "O ponto onde a queda desacelera forma um 'cotovelo' — esse é o K sugerido.<br>"
          "• <b>Silhouette Score</b>: mede o quão bem cada ponto está no seu cluster. "
          "Escolha o K com o maior valor de silhouette.")

    with st.spinner("Carregando Mall Customers..."):
        try:
            df_mall = carregar_mall()
        except Exception as e:
            st.error(f"Erro: {e}"); st.stop()

    df_2d = df_mall[["Annual_Income_(k$)","Spending_Score"]]
    pre = ColumnTransformer([("num",StandardScaler(),df_2d.columns.tolist())])

    with st.spinner("Calculando Elbow e Silhouette para K=2 a 10..."):
        k_range   = range(2, 11)
        inertias  = []
        silhouettes = []
        for k in k_range:
            pipe_k = Pipeline([("pre",pre),
                                ("km", KMeans(n_clusters=k, n_init=10, random_state=42))])
            pipe_k.fit(df_2d)
            inertias.append(pipe_k.named_steps["km"].inertia_)
            silhouettes.append(silhouette_score(df_2d, pipe_k.named_steps["km"].labels_))

    drops = [np.nan] + [1 - inertias[i]/inertias[i-1] for i in range(1,len(inertias))]
    k_elbow = 3
    mask = np.asarray(drops) < 0.1
    if mask.any():
        i_e = max(int(np.argmax(mask)) - 1, 0)
        k_elbow = list(k_range)[i_e]
    k_sil = list(k_range)[int(np.argmax(silhouettes))]
    k_best = min(k_elbow, k_sil)

    caixa(f"<b>✅ K sugerido pelo Elbow:</b> {k_elbow} &nbsp;|&nbsp; "
          f"<b>K sugerido pelo Silhouette:</b> {k_sil} &nbsp;|&nbsp; "
          f"<b>K escolhido (mínimo dos dois):</b> {k_best}",
          cor="#f0fff4", borda="#38a169")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(list(k_range), inertias, marker="o", linewidth=2, color="#667eea")
    axes[0].axvline(k_elbow, linestyle="--", color="tomato",
                    linewidth=1.5, label=f"k_elbow={k_elbow}")
    axes[0].set_xlabel("K"); axes[0].set_ylabel("Inércia (SSE)")
    axes[0].set_title("Método do Cotovelo (Elbow)")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(list(k_range), silhouettes, marker="s", linewidth=2, color="#38a169")
    axes[1].axvline(k_sil, linestyle=":", color="tomato",
                    linewidth=1.5, label=f"k_sil={k_sil}")
    axes[1].set_xlabel("K"); axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Coeficiente de Silhouette")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    save_and_show(fig, "mall_elbow_silhouette")

    divider()
    secao("Métricas de Avaliação Detalhadas")

    df_metricas = pd.DataFrame({
        "K": list(k_range),
        "Inércia": [f"{v:.2f}" for v in inertias],
        "Silhouette": [f"{v:.4f}" for v in silhouettes],
        "Queda Inércia (%)": [f"{(1-drops[i])*100:.1f}%" if not np.isnan(drops[i]) else "—"
                              for i in range(len(k_range))],
    })
    st.dataframe(df_metricas, use_container_width=True)

    divider()
    secao("Resultado com K ideal")
    pipe_best = Pipeline([("pre",pre),
                          ("km", KMeans(n_clusters=k_best, n_init=10, random_state=42))])
    pipe_best.fit(df_2d)
    labels_best   = pipe_best.predict(df_2d)
    centroids_scl = pipe_best.named_steps["km"].cluster_centers_
    scaler_best   = pipe_best.named_steps["pre"].named_transformers_["num"]
    centroids_orig = scaler_best.inverse_transform(centroids_scl)

    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.cm.get_cmap("tab10", k_best)
    ax.scatter(df_2d["Annual_Income_(k$)"], df_2d["Spending_Score"],
               c=labels_best, cmap=cmap, alpha=0.75, s=55)
    ax.scatter(centroids_orig[:,0], centroids_orig[:,1],
               c="red", marker="X", s=250, zorder=5, label="Centroides")
    ax.set_xlabel("Annual Income (k$)"); ax.set_ylabel("Spending Score")
    ax.set_title(f"K-Means (K={k_best}) — K otimizado")
    ax.legend(); ax.grid(alpha=0.3)
    save_and_show(fig, f"mall_kmeans_otimizado_k{k_best}")

    divider()
    with st.expander("Ver código Python — Elbow e Silhouette"):
        code_block("""
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

k_range    = range(2, 11)
inertias   = []
silhouettes = []

for k in k_range:
    pipe = Pipeline([
        ('pre',    pre_kmeans),   # ColumnTransformer já definido
        ('kmeans', KMeans(n_clusters=k, n_init=10, random_state=42))
    ])
    pipe.fit(df_2d)
    inertias.append(pipe.named_steps['kmeans'].inertia_)
    silhouettes.append(silhouette_score(df_2d, pipe.named_steps['kmeans'].labels_))

# Queda relativa da inércia
drops = [np.nan] + [1 - inertias[i]/inertias[i-1] for i in range(1, len(inertias))]

# K sugerido pelo Silhouette (maior valor)
k_sil = list(k_range)[np.argmax(silhouettes)]
print(f"K sugerido (Silhouette): {k_sil}")
""")

# ============================================================================
# PCA
# ============================================================================
elif pagina == "🌀  PCA — Redução de Dimensionalidade":
    st.markdown('<h2>🌀 PCA — Principal Component Analysis</h2>', unsafe_allow_html=True)
    divider()

    caixa("<b>O que é PCA?</b><br>"
          "Técnica de <b>redução de dimensionalidade linear</b>. Transforma os dados "
          "originais em novas variáveis — as <b>componentes principais</b> — que são "
          "combinações lineares das features originais, ordenadas pela variância que "
          "explicam. O primeiro componente captura a maior variação, o segundo a "
          "segunda maior, e assim por diante.")

    divider()
    secao("Conceitos Fundamentais do PCA")
    cp1, cp2 = st.columns(2)
    with cp1:
        caixa("<b>Variância Explicada</b><br>"
              "Cada componente principal explica uma fração da variância total dos dados. "
              "O <b>explained_variance_ratio_</b> informa quanto de informação foi "
              "preservado. Ex: 95% com 2 componentes = compressão com pouca perda.",
              cor="#f0f8ff", borda="#4C72B0")
        caixa("<b>Quando usar PCA?</b><br>"
              "• Quando há muitas features correlacionadas<br>"
              "• Para visualizar dados de alta dimensão em 2D/3D<br>"
              "• Para reduzir custo computacional antes de treinar modelos<br>"
              "• PCA é <b>linear</b> — para padrões não lineares, use t-SNE",
              cor="#f0f8ff", borda="#4C72B0")
    with cp2:
        caixa("<b>⚠️ PCA precisa de escalonamento!</b><br>"
              "O PCA é sensível à escala. Features em escalas maiores "
              "dominam as componentes principais. Sempre aplique "
              "<code>StandardScaler</code> antes do PCA.",
              cor="#fff0f0", borda="#e53e3e")
        caixa("<b>PCA não é supervisionado</b><br>"
              "O PCA não usa os rótulos (<code>y</code>) — é uma transformação "
              "puramente dos dados <code>X</code>. Os componentes principais podem "
              "não separar as classes da forma que você espera — mas geralmente ajudam.",
              cor="#fff8e1", borda="#f59e0b")

    divider()
    secao("PCA no Dataset Breast Cancer (30 features → 2D)")

    with st.spinner("Carregando e aplicando PCA..."):
        df_bc = carregar_breast_cancer_cluster()
        X_bc = df_bc.drop("target", axis=1)
        y_bc = df_bc["target"]

        n_comp = st.slider("Número de componentes PCA:", 2, 10, 2, 1, key="nc_pca")

        pre_pca = ColumnTransformer([("num", StandardScaler(), X_bc.columns.tolist())])
        pipe_pca = Pipeline([("pre", pre_pca), ("pca", PCA(n_components=n_comp, random_state=42))])
        X_pca = pipe_pca.fit_transform(X_bc)
        evr = pipe_pca.named_steps["pca"].explained_variance_ratio_

    col_pca1, col_pca2, col_pca3 = st.columns(3)
    col_pca1.metric("PC1 explica", f"{evr[0]*100:.1f}%")
    col_pca2.metric("PC2 explica", f"{evr[1]*100:.1f}%" if n_comp >= 2 else "—")
    col_pca3.metric("Total explicado", f"{evr[:n_comp].sum()*100:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    paleta = {"benign": "#38a169", "malignant": "#e53e3e"}
    for cls, cor in paleta.items():
        mask = y_bc == cls
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                        color=cor, alpha=0.65, s=25, label=cls)
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    axes[0].set_title("PCA (2 componentes) — Classes Reais")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    pre_full = ColumnTransformer([("num", StandardScaler(), X_bc.columns.tolist())])
    pipe_full = Pipeline([("pre", pre_full),
                          ("pca", PCA(n_components=min(30, X_bc.shape[1]), random_state=42))])
    pipe_full.fit(X_bc)
    evr_all  = pipe_full.named_steps["pca"].explained_variance_ratio_
    evr_cum  = np.cumsum(evr_all)
    axes[1].bar(range(1, len(evr_all)+1), evr_all*100, color="#667eea", alpha=0.7, label="Individual")
    axes[1].plot(range(1, len(evr_cum)+1), evr_cum*100, "r-o", markersize=4,
                 linewidth=2, label="Acumulada")
    axes[1].axhline(95, color="gold", linestyle="--", linewidth=1.5, label="95%")
    axes[1].set_xlabel("Componente"); axes[1].set_ylabel("Variância Explicada (%)")
    axes[1].set_title("Variância Explicada por Componente")
    axes[1].legend(); axes[1].grid(alpha=0.3, axis="y")
    plt.tight_layout()
    save_and_show(fig, "bc_pca_scatter_variance")

    divider()
    secao("PCA + K-Means no Breast Cancer")

    k_pca = st.slider("K para K-Means:", 2, 5, 2, 1, key="k_pca_km")
    pipe_pca_km = Pipeline([
        ("pre",    pre_pca),
        ("pca",    PCA(n_components=2, random_state=42)),
        ("kmeans", KMeans(n_clusters=k_pca, n_init=10, random_state=42)),
    ])
    X_tr_pca = pipe_pca.transform(X_bc)
    df_pca_vis = pd.DataFrame(X_tr_pca[:,:2], columns=["pca0","pca1"])
    pipe_pca_km.fit(X_bc)
    labels_pca  = pipe_pca_km.predict(X_bc)
    cents_pca   = pipe_pca_km.named_steps["kmeans"].cluster_centers_

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cmap = plt.cm.get_cmap("tab10", k_pca)
    axes[0].scatter(df_pca_vis["pca0"], df_pca_vis["pca1"],
                    c=labels_pca, cmap=cmap, alpha=0.7, s=20)
    axes[0].scatter(cents_pca[:,0], cents_pca[:,1],
                    c="red", marker="X", s=150, zorder=5, label="Centroides")
    axes[0].set_xlabel("pca0"); axes[0].set_ylabel("pca1")
    axes[0].set_title(f"PCA + K-Means (K={k_pca})"); axes[0].legend(); axes[0].grid(alpha=0.3)
    y_codes = pd.Categorical(y_bc).codes
    axes[1].scatter(df_pca_vis["pca0"], df_pca_vis["pca1"],
                    c=y_codes, cmap="RdYlGn", alpha=0.7, s=20)
    axes[1].set_xlabel("pca0"); axes[1].set_ylabel("pca1")
    axes[1].set_title("PCA + Classes Reais (y)"); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    save_and_show(fig, f"bc_pca_kmeans_k{k_pca}")

    if k_pca == 2:
        sil_pca = silhouette_score(X_tr_pca[:,:2], labels_pca)
        h_score = homogeneity_score(y_bc, labels_pca)
        c_score = completeness_score(y_bc, labels_pca)
        v_score = v_measure_score(y_bc, labels_pca)
        cm1,cm2,cm3,cm4 = st.columns(4)
        cm1.metric("Silhouette",   f"{sil_pca:.4f}")
        cm2.metric("Homogeneity",  f"{h_score:.4f}")
        cm3.metric("Completeness", f"{c_score:.4f}")
        cm4.metric("V-Measure",    f"{v_score:.4f}")

    divider()
    with st.expander("Ver código Python — PCA + K-Means"):
        code_block("""
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, homogeneity_score, v_measure_score

pre = ColumnTransformer([("num", StandardScaler(), numerical_cols)])

# Pipeline: escalar → PCA → K-Means
pipe = Pipeline([
    ("pre",    pre),
    ("pca",    PCA(n_components=2, random_state=42)),
    ("kmeans", KMeans(n_clusters=2, n_init=10, random_state=42))
])
pipe.fit(X)

# Variância explicada
evr = pipe.named_steps['pca'].explained_variance_ratio_
print(f"PC1: {evr[0]*100:.1f}% | PC2: {evr[1]*100:.1f}%")
print(f"Total: {evr.sum()*100:.1f}%")

labels = pipe.predict(X)
""")

# ============================================================================
# EXERCÍCIO — BREAST CANCER
# ============================================================================
elif pagina == "💊  Exercício — Breast Cancer":
    st.markdown('<h2>💊 Exercício — PCA + K-Means no Breast Cancer</h2>',
                unsafe_allow_html=True)
    divider()

    caixa("<b>Desafio</b><br>"
          "Aplicar PCA e K-Means ao dataset Breast Cancer (30 features) e comparar "
          "os clusters encontrados com os rótulos reais (benign/malignant). "
          "Avaliar com Silhouette, Homogeneity, Completeness e V-Measure.")

    with st.spinner("Carregando e processando..."):
        df_bc = carregar_breast_cancer_cluster()
        X_bc = df_bc.drop("target", axis=1); y_bc = df_bc["target"]

    st.markdown(f"**Dataset:** {df_bc.shape[0]} amostras × {df_bc.shape[1]} colunas")

    secao("Pré-processamento e PCA")
    n_comp_ex = st.slider("Componentes PCA:", 2, 10, 2, 1, key="nce")

    pre_bc = ColumnTransformer([("num", StandardScaler(), X_bc.columns.tolist())])
    pipe_pca_ex = Pipeline([("pre", pre_bc),
                             ("pca", PCA(n_components=n_comp_ex, random_state=42))])
    X_pca_ex = pipe_pca_ex.fit_transform(X_bc)
    evr_ex = pipe_pca_ex.named_steps["pca"].explained_variance_ratio_

    ce1, ce2, ce3 = st.columns(3)
    ce1.metric("PC1", f"{evr_ex[0]*100:.1f}%")
    ce2.metric("PC2", f"{evr_ex[1]*100:.1f}%" if n_comp_ex >= 2 else "—")
    ce3.metric(f"Total ({n_comp_ex} PCs)", f"{evr_ex.sum()*100:.1f}%")

    divider()
    secao("K-Means sobre as componentes PCA")
    k_ex = st.slider("K:", 2, 5, 2, 1, key="k_ex")

    km_ex = KMeans(n_clusters=k_ex, n_init=10, random_state=42)
    labels_ex = km_ex.fit_predict(X_pca_ex[:,:2])
    cents_ex  = km_ex.cluster_centers_

    sil_ex = silhouette_score(X_pca_ex[:,:2], labels_ex)
    h_ex   = homogeneity_score(y_bc, labels_ex)
    c_ex   = completeness_score(y_bc, labels_ex)
    v_ex   = v_measure_score(y_bc, labels_ex)

    ce1, ce2, ce3, ce4 = st.columns(4)
    ce1.metric("Silhouette",   f"{sil_ex:.4f}")
    ce2.metric("Homogeneity",  f"{h_ex:.4f}")
    ce3.metric("Completeness", f"{c_ex:.4f}")
    ce4.metric("V-Measure",    f"{v_ex:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    cmap = plt.cm.get_cmap("tab10", k_ex)
    sc0 = axes[0].scatter(X_pca_ex[:,0], X_pca_ex[:,1],
                          c=labels_ex, cmap=cmap, alpha=0.7, s=20)
    axes[0].scatter(cents_ex[:,0], cents_ex[:,1], c="red",
                    marker="X", s=150, zorder=5, label="Centroides")
    axes[0].set_xlabel("pca0"); axes[0].set_ylabel("pca1")
    axes[0].set_title(f"K-Means (K={k_ex}) sobre PCA")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    y_codes = pd.Categorical(y_bc).codes
    sc1 = axes[1].scatter(X_pca_ex[:,0], X_pca_ex[:,1],
                          c=y_codes, cmap="RdYlGn", alpha=0.7, s=20)
    axes[1].set_xlabel("pca0"); axes[1].set_ylabel("pca1")
    axes[1].set_title("Classes Reais (benign=verde, malignant=vermelho)")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    save_and_show(fig, "bc_exercicio_pca_kmeans")

    divider()
    with st.expander("Ver código Python — Exercício completo"):
        code_block("""
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    silhouette_score, homogeneity_score,
    completeness_score, v_measure_score
)
import pandas as pd
import matplotlib.pyplot as plt

data = load_breast_cancer()
df   = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target_names[data.target]

X = df.drop("target", axis=1)
y = df["target"]

pre = ColumnTransformer([("num", StandardScaler(), X.columns.tolist())])
pipe = Pipeline([
    ("pre",    pre),
    ("pca",    PCA(n_components=2, random_state=42)),
    ("kmeans", KMeans(n_clusters=2, n_init=10, random_state=42))
])
pipe.fit(X)

X_pca  = pipe.named_steps['pca'].transform(
             pipe.named_steps['pre'].transform(X))
labels = pipe.predict(X)
cents  = pipe.named_steps['kmeans'].cluster_centers_

print(f"Silhouette:   {silhouette_score(X_pca, labels):.4f}")
print(f"Homogeneity:  {homogeneity_score(y, labels):.4f}")
print(f"V-Measure:    {v_measure_score(y, labels):.4f}")
""")

    divider()
    st.markdown("""<div class="footer">
        <p style="margin:0;font-size:1.1rem;font-weight:700;">Machine Learning — Aula 07</p>
        <p style="margin:.3rem 0 0;font-size:.9rem;opacity:.8;">
        K-Means e PCA · Cláudio Ferreira Neves · SENAI/SC</p></div>""",
        unsafe_allow_html=True)
