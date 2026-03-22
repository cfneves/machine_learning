"""
=============================================================================
Machine Learning — Aula 03
Regressão Linear e Polinomial
Aplicação Streamlit interativa — material didático para iniciantes

Autor       : Cláudio Ferreira Neves
Cargo atual : Analista de BI — Save Co. | Jaraguá do Sul/SC
Docência    : Especialista de Ensino II — Análise de Dados | SENAI/SC
Certificação: DATA ANALYST CERTIFIED PROFESSIONAL © (DACP)
=============================================================================
"""

import os
import numpy as np

# ---------------------------------------------------------------------------
# Backend Agg: deve ser definido ANTES de qualquer import de matplotlib.pyplot
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# URLs de navegação entre aulas
# ---------------------------------------------------------------------------
PAGE_PORTAL  = "pages/Portal.py"
PAGE_AULA_02 = "pages/Aula_02.py"
PAGE_AULA_04 = "pages/Aula_04.py"

import seaborn as sns
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import (
    PolynomialFeatures,
    StandardScaler,
    OneHotEncoder,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import fetch_california_housing

# ---------------------------------------------------------------------------
# Pasta de saída para salvar os gráficos gerados
# ---------------------------------------------------------------------------
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def save_and_show(fig: plt.Figure, filename: str):
    """Salva a figura em outputs/ e exibe no Streamlit."""
    path = os.path.join(OUTPUTS_DIR, f"{filename}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    st.pyplot(fig)
    plt.close(fig)


def caixa_conceito(html: str):
    st.markdown(f'<div class="concept-box">{html}</div>', unsafe_allow_html=True)


def caixa_dica(html: str):
    st.markdown(f'<div class="tip-box">💡 {html}</div>', unsafe_allow_html=True)


def caixa_atencao(html: str):
    st.markdown(f'<div class="warn-box">⚠️ {html}</div>', unsafe_allow_html=True)


def secao(texto: str):
    st.markdown(f"### {texto}")


def divider():
    st.markdown("---")


def code_block(code: str, title: str = ""):
    if title:
        st.markdown(f"**{title}**")
    st.code(code, language="python")


# ============================================================================
# CARREGAMENTO DE DADOS (cache para performance)
# ============================================================================

@st.cache_data
def carregar_insurance() -> pd.DataFrame:
    """Carrega o dataset Insurance do GitHub."""
    url = (
        "https://raw.githubusercontent.com/matheusvanzan/"
        "Machine-Learning-Examples/refs/heads/master/datasets/insurance.csv"
    )
    df = pd.read_csv(url)
    return df


@st.cache_data
def carregar_california() -> tuple:
    """Carrega o dataset California Housing do scikit-learn."""
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    return df, housing.feature_names, housing.target_names


# ============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="Regressão Linear e Polinomial",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS — visual premium (mesmo padrão da aula_01 e aula_02)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

.sidebar-header {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    padding: 1.2rem; border-radius: 10px;
    text-align: center; margin-bottom: 1rem; color: white;
}

.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px; padding: 1rem 1.5rem; color: white;
    text-align: center; margin-bottom: 0.5rem;
    box-shadow: 0 4px 15px rgba(102,126,234,0.4);
}
.metric-card h2 { margin: 0; font-size: 2rem; }
.metric-card p  { margin: 0; opacity: 0.85; font-size: 0.9rem; }

.concept-box {
    background: #f0f4ff; border-left: 5px solid #667eea;
    padding: 1rem 1.5rem; border-radius: 0 10px 10px 0; margin: 0.8rem 0;
}

.tip-box {
    background: #fff8e1; border-left: 5px solid #f59e0b;
    padding: 1rem 1.5rem; border-radius: 0 10px 10px 0; margin: 0.8rem 0;
}

.warn-box {
    background: #fff0f0; border-left: 5px solid #e53e3e;
    padding: 1rem 1.5rem; border-radius: 0 10px 10px 0; margin: 0.8rem 0;
}

.explain-box {
    background: #f0fff4; border-left: 5px solid #38a169;
    padding: 1rem 1.5rem; border-radius: 0 10px 10px 0; margin: 0.8rem 0;
}

.section-title { font-size: 1.6rem; font-weight: 700; color: #1a1a2e; }

.footer {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    color: white; padding: 2rem; border-radius: 12px;
    text-align: center; margin-top: 3rem;
}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h2 style="margin:0; font-size:1.3rem;">📈 Regressão</h2>
        <p style="margin:0.4rem 0 0; font-size:0.8rem; opacity:0.8;">Linear e Polinomial</p>
    </div>
    """, unsafe_allow_html=True)

    pagina = st.radio(
        "Navegação",
        options=[
            "🏠  Início",
            "📐  Regressão Linear Simples",
            "🔢  Regressão Linear Múltipla",
            "📉  Gradiente Descendente",
            "🔵  Regressão Polinomial",
            "🏘️  Dataset Real — Imóveis (California Housing)",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#888; text-align:center;'>
        📁 Gráficos salvos em <code>outputs/</code>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# AUTOR — exibido em todas as páginas
# ============================================================================

# ── navegação entre aulas ────────────────────────────────────────────────────
from nav import tab_nav
tab_nav(3)  # replace CURRENT_NUMBER with the correct int

st.markdown(
    "<p style='text-align:center; font-size:0.95rem; color:#667eea; "
    "font-weight:600; margin-bottom:0;'>"
    "Cláudio Ferreira Neves &nbsp;·&nbsp; Especialista em Ciência de Dados e IA"
    "</p>",
    unsafe_allow_html=True,
)

_nav_l, _nav_m, _nav_r = st.columns([1.2, 4, 1.2])
with _nav_l:
    if st.button("← Aula 02", use_container_width=True, key="nav_prev"):
        st.switch_page(PAGE_AULA_02)
with _nav_m:
    if st.button("🏠 Portal", use_container_width=True, key="nav_portal"):
        st.switch_page(PAGE_PORTAL)
with _nav_r:
    if st.button("Aula 04 →", use_container_width=True, key="nav_next"):
        st.switch_page(PAGE_AULA_04)


# ============================================================================
# PÁGINA: INÍCIO
# ============================================================================

if pagina == "🏠  Início":

    st.markdown("""
    <div style='text-align:center; padding: 2rem 0 1rem;'>
        <h1 style='font-size:2.8rem; font-weight:800; color:#1a1a2e; margin-bottom:0.3rem;'>
            📈 Regressão Linear e Polinomial
        </h1>
        <p style='font-size:1.15rem; color:#555; max-width:720px; margin:0 auto;'>
            Do imóvel ao seguro de saúde — modelos que preveem números.
            Aprenda como ajustar retas (e curvas) a dados reais com scikit-learn.
        </p>
    </div>
    """, unsafe_allow_html=True)

    divider()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><h2>3</h2><p>Datasets</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h2>📐</h2><p>Linear Simples</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h2>🔵</h2><p>Polinomial</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><h2>⚙️</h2><p>Pipeline</p></div>', unsafe_allow_html=True)

    divider()

    st.markdown("## 📚 O que você vai aprender")
    col_a, col_b = st.columns(2)

    with col_a:
        caixa_conceito("""
        <b>📐 Regressão Linear Simples</b><br>
        Modelar a relação entre uma variável preditora e um alvo contínuo.
        Fórmula <code>ŷ = β₀ + β₁x</code>, coeficientes e métricas de avaliação.
        """)
        caixa_conceito("""
        <b>🔢 Regressão Linear Múltipla</b><br>
        Estender o modelo para múltiplas variáveis preditoras.
        Pré-processamento com Pipeline + ColumnTransformer no dataset Insurance.
        """)
        caixa_conceito("""
        <b>📉 Gradiente Descendente</b><br>
        Compreender como o modelo aprende iterativamente minimizando o erro.
        Taxa de aprendizado, convergência e SGDRegressor na prática.
        """)

    with col_b:
        caixa_conceito("""
        <b>🔵 Regressão Polinomial</b><br>
        Capturar relações não-lineares com <code>PolynomialFeatures</code>.
        Comparar graus e identificar sobreajuste (overfitting).
        """)
        caixa_conceito("""
        <b>🏘️ California Housing</b><br>
        Aplicar regressão linear e polinomial em um dataset real com 20640 imóveis
        do estado da Califórnia. Pipeline completo com StandardScaler.
        """)
        caixa_conceito("""
        <b>⚙️ Pipeline scikit-learn</b><br>
        Encadear pré-processamento e modelo em um único objeto reutilizável —
        o padrão de produção em projetos de ML.
        """)

    divider()
    st.markdown("## 🧰 Tecnologias utilizadas")
    cols = st.columns(5)
    for col, lib in zip(cols, ["NumPy", "Pandas", "Matplotlib", "Seaborn", "scikit-learn"]):
        with col:
            st.info(f"**{lib}**")

    divider()
    caixa_dica("""
    <b>Como usar esta aplicação:</b> use o menu lateral para navegar entre os tópicos.
    Cada seção contém a <b>teoria explicada</b>, <b>controles interativos</b>,
    <b>visualizações comentadas</b> e o <b>código documentado linha a linha</b>.
    Todos os gráficos são salvos automaticamente na pasta <code>outputs/</code>.
    """)


# ============================================================================
# PÁGINA: REGRESSÃO LINEAR SIMPLES
# ============================================================================

elif pagina == "📐  Regressão Linear Simples":

    st.markdown("""
    <div style='text-align:center; padding: 1.5rem 0 0.5rem;'>
        <h1 style='font-size:2.4rem; font-weight:800; color:#1a1a2e;'>
            📐 Regressão Linear Simples
        </h1>
        <p style='font-size:1.05rem; color:#555;'>
            Uma variável preditora → uma variável alvo contínua
        </p>
    </div>
    """, unsafe_allow_html=True)

    divider()

    caixa_conceito("""
    <b>Fórmula:</b> <code>ŷ = β₀ + β₁x</code><br><br>
    Onde:<br>
    &nbsp;&nbsp;• <b>ŷ</b> = valor previsto (preço do imóvel)<br>
    &nbsp;&nbsp;• <b>β₀</b> = intercepto (preço base, sem metragem)<br>
    &nbsp;&nbsp;• <b>β₁</b> = coeficiente angular (quanto o preço aumenta por m²)<br>
    &nbsp;&nbsp;• <b>x</b> = variável preditora (metragem em m²)<br><br>
    <b>Analogia imobiliária:</b> a cada m² adicional, o preço sobe β₁ unidades.
    O modelo encontra a reta que minimiza a soma dos erros ao quadrado.
    """)

    divider()

    secao("⚙️ Parâmetros do experimento")

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        n_samples = st.slider("Número de amostras", min_value=50, max_value=300,
                              value=100, step=10)
    with col_s2:
        noise_sigma = st.slider("Ruído σ", min_value=0, max_value=50,
                                value=20, step=5)
    with col_s3:
        x_new = st.slider("Metragem para previsão (m²)", min_value=40, max_value=100,
                          value=75, step=1)

    # Gerar dados artificiais
    np.random.seed(42)
    X_lin = np.linspace(40, 100, n_samples).reshape(-1, 1)
    y_lin = 3.5 * X_lin.flatten() + 20 + np.random.normal(0, noise_sigma, n_samples)

    # Treinar modelo
    X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
        X_lin, y_lin, test_size=0.2, random_state=42
    )
    model_lin = LinearRegression()
    model_lin.fit(X_train_l, y_train_l)
    y_pred_l = model_lin.predict(X_test_l)

    b0 = model_lin.intercept_
    b1 = model_lin.coef_[0]
    r2 = r2_score(y_test_l, y_pred_l)
    rmse = np.sqrt(mean_squared_error(y_test_l, y_pred_l))
    mae = mean_absolute_error(y_test_l, y_pred_l)
    pred_new = model_lin.predict([[x_new]])[0]

    divider()
    secao("📊 Métricas do modelo")

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    with mc1:
        st.markdown(f'<div class="metric-card"><h2>{b0:.1f}</h2><p>β₀ (intercepto)</p></div>',
                    unsafe_allow_html=True)
    with mc2:
        st.markdown(f'<div class="metric-card"><h2>{b1:.2f}</h2><p>β₁ (coeficiente)</p></div>',
                    unsafe_allow_html=True)
    with mc3:
        st.markdown(f'<div class="metric-card"><h2>{r2:.3f}</h2><p>R²</p></div>',
                    unsafe_allow_html=True)
    with mc4:
        st.markdown(f'<div class="metric-card"><h2>{rmse:.1f}</h2><p>RMSE</p></div>',
                    unsafe_allow_html=True)
    with mc5:
        st.markdown(f'<div class="metric-card"><h2>{mae:.1f}</h2><p>MAE</p></div>',
                    unsafe_allow_html=True)

    st.markdown(
        f"<div class='explain-box'><b>Previsão para {x_new} m²:</b> "
        f"ŷ = {b0:.1f} + {b1:.2f} × {x_new} = <b>{pred_new:.1f}</b> (unidades de preço)</div>",
        unsafe_allow_html=True,
    )

    divider()

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        secao("📈 Dados + Reta Ajustada")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(X_lin, y_lin, alpha=0.5, color="#667eea", s=25, label="Dados")
        x_range = np.linspace(40, 100, 200).reshape(-1, 1)
        ax.plot(x_range, model_lin.predict(x_range), color="#e53e3e",
                linewidth=2, label=f"ŷ = {b0:.1f} + {b1:.2f}x")
        ax.scatter([x_new], [pred_new], color="#f59e0b", s=200, zorder=5,
                   marker="*", label=f"Previsão: {pred_new:.1f}")
        ax.set_xlabel("Metragem (m²)")
        ax.set_ylabel("Preço")
        ax.set_title("Regressão Linear Simples — Imóveis", fontweight="bold")
        ax.legend(fontsize=8)
        fig.tight_layout()
        save_and_show(fig, "lin_simples_scatter")

    with col_p2:
        secao("🎯 Gráfico de Paridade (Real vs Previsto)")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(y_test_l, y_pred_l, alpha=0.6, color="#764ba2", s=35)
        lims = [min(y_test_l.min(), y_pred_l.min()),
                max(y_test_l.max(), y_pred_l.max())]
        ax.plot(lims, lims, "k--", linewidth=1.5, label="Previsão perfeita")
        ax.set_xlabel("Valor Real")
        ax.set_ylabel("Valor Previsto")
        ax.set_title("Paridade: Real vs Previsto", fontweight="bold")
        ax.legend(fontsize=8)
        fig.tight_layout()
        save_and_show(fig, "lin_simples_paridade")

    divider()

    secao("📖 Entendendo as métricas")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        caixa_conceito("""
        <b>R² — Coeficiente de Determinação</b><br>
        Mede a proporção da variância do alvo explicada pelo modelo.
        Varia de 0 a 1 (pode ser negativo para modelos ruins).<br>
        &nbsp;&nbsp;• R² = 1.0 → previsão perfeita<br>
        &nbsp;&nbsp;• R² = 0.0 → modelo não explica nada<br>
        &nbsp;&nbsp;• R² = 0.8 → modelo explica 80% da variância
        """)
        caixa_conceito("""
        <b>MAE — Mean Absolute Error</b><br>
        Média dos erros absolutos: <code>MAE = Σ|yᵢ - ŷᵢ| / n</code><br>
        Fácil de interpretar: erro médio na mesma unidade do alvo.
        Menos sensível a outliers do que MSE.
        """)
    with col_m2:
        caixa_conceito("""
        <b>MSE — Mean Squared Error</b><br>
        Média dos erros ao quadrado: <code>MSE = Σ(yᵢ - ŷᵢ)² / n</code><br>
        Penaliza erros grandes mais fortemente. Unidade = alvo².
        É a função de custo que o LinearRegression minimiza.
        """)
        caixa_conceito("""
        <b>RMSE — Root Mean Squared Error</b><br>
        Raiz quadrada do MSE: <code>RMSE = √MSE</code><br>
        Mesma unidade do alvo — mais interpretável que MSE.
        Ainda penaliza erros grandes. Métrica padrão em regressão.
        """)

    divider()

    with st.expander("📝 Ver código desta seção"):
        code_block("""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

np.random.seed(42)
n_samples = 100

# Gerar dados artificiais: metragem de 40 a 100 m²
# Preço = 3.5 * metragem + 20 + ruído gaussiano
X = np.linspace(40, 100, n_samples).reshape(-1, 1)
y = 3.5 * X.flatten() + 20 + np.random.normal(0, 20, n_samples)

# Dividir em treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Treinar o modelo de Regressão Linear
model = LinearRegression()
model.fit(X_train, y_train)

# Coeficientes aprendidos
beta_0 = model.intercept_   # intercepto
beta_1 = model.coef_[0]     # coeficiente angular
print(f"ŷ = {beta_0:.2f} + {beta_1:.2f} * x")

# Avaliar no conjunto de teste
y_pred = model.predict(X_test)
r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
print(f"R²={r2:.4f}  RMSE={rmse:.2f}  MAE={mae:.2f}")

# Previsão para um novo imóvel de 75 m²
x_novo = np.array([[75]])
preco_previsto = model.predict(x_novo)[0]
print(f"Previsão para 75m²: {preco_previsto:.2f}")

# Gráfico
fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(X, y, alpha=0.5, color='steelblue', label='Dados')
x_range = np.linspace(40, 100, 200).reshape(-1, 1)
ax.plot(x_range, model.predict(x_range), 'r-', linewidth=2, label='Reta ajustada')
ax.scatter([75], [preco_previsto], color='orange', s=200, marker='*', label='Previsão')
ax.set_xlabel("Metragem (m²)")
ax.set_ylabel("Preço")
ax.legend()
plt.tight_layout()
plt.show()
""")

    st.markdown("""
    <div class="footer">
        <p style="margin:0; font-size:1rem; font-weight:600;">
            Machine Learning — Aula 03 · Regressão Linear Simples
        </p>
        <p style="margin:0.3rem 0 0; font-size:0.85rem; opacity:0.7;">
            Gráficos salvos em <code>outputs/</code>
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# PÁGINA: REGRESSÃO LINEAR MÚLTIPLA
# ============================================================================

elif pagina == "🔢  Regressão Linear Múltipla":

    st.markdown("""
    <div style='text-align:center; padding: 1.5rem 0 0.5rem;'>
        <h1 style='font-size:2.4rem; font-weight:800; color:#1a1a2e;'>
            🔢 Regressão Linear Múltipla
        </h1>
        <p style='font-size:1.05rem; color:#555;'>
            Múltiplas variáveis preditoras para um alvo contínuo
        </p>
    </div>
    """, unsafe_allow_html=True)

    divider()

    caixa_conceito("""
    <b>Fórmula:</b> <code>ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ</code><br><br>
    Cada variável preditora xᵢ contribui com seu coeficiente βᵢ para a previsão final.
    O modelo encontra todos os coeficientes simultaneamente minimizando o MSE.
    Com múltiplas variáveis, é essencial fazer <b>pré-processamento</b> correto
    (codificação de variáveis categóricas, escalonamento) antes de treinar.
    """)

    divider()

    tab_artificial, tab_insurance = st.tabs(["🏠 Dados Artificiais", "🏥 Dataset Insurance"])

    # ------------------------------------------------------------------
    # TAB 1: DADOS ARTIFICIAIS
    # ------------------------------------------------------------------
    with tab_artificial:

        secao("📐 Dataset Artificial — Preço de Imóveis")

        caixa_dica("""
        Geramos imóveis artificiais com <b>metragem</b> (40–200 m²) e
        <b>número de quartos</b> (1–4). O preço segue a equação:<br>
        <code>preço = 3.2 × metragem + 50 × quartos + 30 + ruído</code>
        """)

        np.random.seed(0)
        n_imoveis = 200
        metragem = np.random.uniform(40, 200, n_imoveis)
        quartos = np.random.randint(1, 5, n_imoveis).astype(float)
        preco = 3.2 * metragem + 50 * quartos + 30 + np.random.normal(0, 30, n_imoveis)

        df_imoveis = pd.DataFrame({
            "metragem": metragem,
            "quartos": quartos,
            "preco": preco,
        })

        st.markdown("**Primeiras 5 linhas do dataset:**")
        st.dataframe(df_imoveis.head(), use_container_width=True)

        col_art1, col_art2 = st.columns(2)
        with col_art1:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(df_imoveis["metragem"], df_imoveis["preco"],
                       alpha=0.5, color="#667eea", s=20)
            ax.set_xlabel("Metragem (m²)")
            ax.set_ylabel("Preço")
            ax.set_title("Metragem × Preço", fontweight="bold")
            fig.tight_layout()
            save_and_show(fig, "mult_scatter_metragem")

        with col_art2:
            fig, ax = plt.subplots(figsize=(5, 4))
            for q in sorted(df_imoveis["quartos"].unique()):
                subset = df_imoveis[df_imoveis["quartos"] == q]
                ax.scatter(subset["quartos"], subset["preco"],
                           alpha=0.5, s=20, label=f"{int(q)} quartos")
            ax.set_xlabel("Número de Quartos")
            ax.set_ylabel("Preço")
            ax.set_title("Quartos × Preço", fontweight="bold")
            ax.legend(fontsize=8)
            fig.tight_layout()
            save_and_show(fig, "mult_scatter_quartos")

        # Treinar modelo
        X_mult = df_imoveis[["metragem", "quartos"]].values
        y_mult = df_imoveis["preco"].values
        X_tr_m, X_te_m, y_tr_m, y_te_m = train_test_split(
            X_mult, y_mult, test_size=0.2, random_state=42
        )
        modelo_mult = LinearRegression()
        modelo_mult.fit(X_tr_m, y_tr_m)
        y_pred_m = modelo_mult.predict(X_te_m)

        b0_m = modelo_mult.intercept_
        b1_m, b2_m = modelo_mult.coef_
        r2_m = r2_score(y_te_m, y_pred_m)
        mae_m = mean_absolute_error(y_te_m, y_pred_m)
        rmse_m = np.sqrt(mean_squared_error(y_te_m, y_pred_m))

        st.markdown(
            f"<div class='explain-box'><b>Equação aprendida:</b><br>"
            f"ŷ = {b0_m:.1f} + {b1_m:.2f} × metragem + {b2_m:.2f} × quartos</div>",
            unsafe_allow_html=True,
        )

        divider()
        secao("⚙️ Previsão interativa")
        col_sl1, col_sl2 = st.columns(2)
        with col_sl1:
            met_input = st.slider("Metragem do imóvel (m²)", 40, 200, 100)
        with col_sl2:
            qrt_input = st.slider("Número de quartos", 1, 4, 2)

        preco_pred = modelo_mult.predict([[met_input, qrt_input]])[0]
        st.markdown(
            f"<div class='explain-box'>Imóvel de <b>{met_input} m²</b> com "
            f"<b>{qrt_input} quartos</b> → preço previsto: <b>{preco_pred:.1f}</b></div>",
            unsafe_allow_html=True,
        )

        divider()
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.markdown(f'<div class="metric-card"><h2>{r2_m:.3f}</h2><p>R²</p></div>',
                        unsafe_allow_html=True)
        with mc2:
            st.markdown(f'<div class="metric-card"><h2>{mae_m:.1f}</h2><p>MAE</p></div>',
                        unsafe_allow_html=True)
        with mc3:
            st.markdown(f'<div class="metric-card"><h2>{rmse_m:.1f}</h2><p>RMSE</p></div>',
                        unsafe_allow_html=True)

        secao("🎯 Gráfico de Paridade")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(y_te_m, y_pred_m, alpha=0.6, color="#764ba2", s=35)
        lims = [min(y_te_m.min(), y_pred_m.min()), max(y_te_m.max(), y_pred_m.max())]
        ax.plot(lims, lims, "k--", linewidth=1.5, label="Perfeito")
        ax.set_xlabel("Valor Real")
        ax.set_ylabel("Valor Previsto")
        ax.set_title("Paridade — Dados Artificiais", fontweight="bold")
        ax.legend(fontsize=8)
        fig.tight_layout()
        save_and_show(fig, "mult_paridade_artificial")

    # ------------------------------------------------------------------
    # TAB 2: INSURANCE
    # ------------------------------------------------------------------
    with tab_insurance:

        secao("🏥 Dataset Insurance — Custo de Seguro de Saúde")

        caixa_conceito("""
        <b>Dataset:</b> informações de segurados americanos com as seguintes colunas:<br>
        &nbsp;&nbsp;• <b>age</b>: idade do segurado<br>
        &nbsp;&nbsp;• <b>sex</b>: sexo (male/female)<br>
        &nbsp;&nbsp;• <b>bmi</b>: índice de massa corporal<br>
        &nbsp;&nbsp;• <b>children</b>: número de dependentes<br>
        &nbsp;&nbsp;• <b>smoker</b>: fumante (yes/no)<br>
        &nbsp;&nbsp;• <b>region</b>: região dos EUA<br>
        &nbsp;&nbsp;• <b>charges</b>: <b>custo anual do seguro (alvo)</b>
        """)

        try:
            df_ins = carregar_insurance()

            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.markdown(f"**Shape:** {df_ins.shape[0]} linhas × {df_ins.shape[1]} colunas")
                st.dataframe(df_ins.head(), use_container_width=True)
            with col_info2:
                st.markdown("**Estatísticas descritivas:**")
                st.dataframe(df_ins.describe().round(2), use_container_width=True)

            divider()
            secao("📊 EDA — Distribuição de charges por fumante")
            fig, ax = plt.subplots(figsize=(7, 4))
            for smk, color in zip(["yes", "no"], ["#e53e3e", "#667eea"]):
                subset = df_ins[df_ins["smoker"] == smk]["charges"]
                ax.hist(subset, bins=30, alpha=0.6, color=color, label=f"Fumante: {smk}")
            ax.set_xlabel("Charges (USD)")
            ax.set_ylabel("Frequência")
            ax.set_title("Distribuição de Charges por Hábito de Fumar", fontweight="bold")
            ax.legend()
            fig.tight_layout()
            save_and_show(fig, "insurance_hist_charges_smoker")

            divider()
            secao("🔧 Pré-processamento + Pipeline")

            # Limpeza
            df_ins = df_ins.dropna().drop_duplicates().reset_index(drop=True)

            num_features = ["age", "bmi", "children"]
            cat_features = ["sex", "smoker", "region"]
            target = "charges"

            X_ins = df_ins[num_features + cat_features]
            y_ins = df_ins[target].values

            X_tr_ins, X_te_ins, y_tr_ins, y_te_ins = train_test_split(
                X_ins, y_ins, test_size=0.2, random_state=42
            )

            # Pré-processador
            preprocessor = ColumnTransformer(transformers=[
                ("num", StandardScaler(), num_features),
                ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_features),
            ])

            # Pipeline
            pipe_ins = Pipeline([
                ("preprocessor", preprocessor),
                ("model", LinearRegression()),
            ])
            pipe_ins.fit(X_tr_ins, y_tr_ins)
            y_pred_ins = pipe_ins.predict(X_te_ins)

            r2_ins   = r2_score(y_te_ins, y_pred_ins)
            mae_ins  = mean_absolute_error(y_te_ins, y_pred_ins)
            rmse_ins = np.sqrt(mean_squared_error(y_te_ins, y_pred_ins))

            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.markdown(f'<div class="metric-card"><h2>{r2_ins:.3f}</h2><p>R²</p></div>',
                            unsafe_allow_html=True)
            with mc2:
                st.markdown(f'<div class="metric-card"><h2>${mae_ins:,.0f}</h2><p>MAE</p></div>',
                            unsafe_allow_html=True)
            with mc3:
                st.markdown(f'<div class="metric-card"><h2>${rmse_ins:,.0f}</h2><p>RMSE</p></div>',
                            unsafe_allow_html=True)

            divider()
            secao("🎯 Gráfico de Paridade — Insurance")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(y_te_ins, y_pred_ins, alpha=0.5, color="#667eea", s=25)
            lims = [0, max(y_te_ins.max(), y_pred_ins.max())]
            ax.plot(lims, lims, "k--", linewidth=1.5, label="Perfeito")
            ax.set_xlabel("Charges Real (USD)")
            ax.set_ylabel("Charges Previsto (USD)")
            ax.set_title("Paridade — Dataset Insurance", fontweight="bold")
            ax.legend(fontsize=8)
            fig.tight_layout()
            save_and_show(fig, "insurance_paridade")

            caixa_atencao("""
            <b>Atenção — Data Leakage:</b> o StandardScaler deve ser ajustado
            (<code>fit</code>) <b>apenas no conjunto de treino</b> e aplicado
            (<code>transform</code>) no conjunto de teste. O Pipeline do scikit-learn
            garante isso automaticamente — nunca chame <code>fit_transform</code>
            no dataset completo antes de dividir!
            """)

            with st.expander("📝 Ver código desta seção"):
                code_block("""
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

# Carregar dados
url = ("https://raw.githubusercontent.com/matheusvanzan/"
       "Machine-Learning-Examples/refs/heads/master/datasets/insurance.csv")
df = pd.read_csv(url).dropna().drop_duplicates().reset_index(drop=True)

num_features = ["age", "bmi", "children"]
cat_features = ["sex", "smoker", "region"]

X = df[num_features + cat_features]
y = df["charges"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pré-processador: escalar numéricas, codificar categóricas
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_features),
])

# Pipeline: pré-processamento + modelo
pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression()),
])

pipe.fit(X_train, y_train)       # fit SOMENTE no treino
y_pred = pipe.predict(X_test)    # transform automático no teste

r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
print(f"R²={r2:.4f}  RMSE={rmse:.2f}  MAE={mae:.2f}")
""")

        except Exception as e:
            caixa_atencao(f"Não foi possível carregar o dataset Insurance. Verifique sua conexão com a internet.<br><code>{e}</code>")

    st.markdown("""
    <div class="footer">
        <p style="margin:0; font-size:1rem; font-weight:600;">
            Machine Learning — Aula 03 · Regressão Linear Múltipla
        </p>
        <p style="margin:0.3rem 0 0; font-size:0.85rem; opacity:0.7;">
            Gráficos salvos em <code>outputs/</code>
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# PÁGINA: GRADIENTE DESCENDENTE
# ============================================================================

elif pagina == "📉  Gradiente Descendente":

    st.markdown("""
    <div style='text-align:center; padding: 1.5rem 0 0.5rem;'>
        <h1 style='font-size:2.4rem; font-weight:800; color:#1a1a2e;'>
            📉 Gradiente Descendente
        </h1>
        <p style='font-size:1.05rem; color:#555;'>
            Como o modelo aprende iterativamente minimizando o erro
        </p>
    </div>
    """, unsafe_allow_html=True)

    divider()

    caixa_conceito("""
    <b>Intuição do Gradiente Descendente:</b><br>
    Imagine estar numa montanha com neblina e querer chegar ao vale (mínimo da função de custo).
    Você dá passos na direção mais íngreme para baixo — o tamanho do passo é a
    <b>taxa de aprendizado (η)</b>.<br><br>
    <b>Algoritmo:</b><br>
    &nbsp;&nbsp;1. Comece com coeficientes aleatórios<br>
    &nbsp;&nbsp;2. Calcule o gradiente da função de custo (derivada parcial)<br>
    &nbsp;&nbsp;3. Atualize: <code>β ← β − η × ∇J(β)</code><br>
    &nbsp;&nbsp;4. Repita por N épocas até convergir
    """)

    divider()
    secao("📈 Visualização da Função de Custo")

    # Função de custo didática: J(β) = (β - 2)² + 1
    beta_vals = np.linspace(-2, 6, 300)
    J_vals = (beta_vals - 2) ** 2 + 1

    col_gd1, col_gd2 = st.columns(2)

    with col_gd1:
        st.markdown("**Passos pequenos (η = 0.1) — convergência suave**")
        eta_small = 0.1
        beta_t = -1.5
        steps_small = [beta_t]
        for _ in range(15):
            grad = 2 * (beta_t - 2)
            beta_t = beta_t - eta_small * grad
            steps_small.append(beta_t)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(beta_vals, J_vals, color="#667eea", linewidth=2.5, label="J(β) = (β-2)²+1")
        for s in steps_small:
            ax.scatter([s], [(s - 2) ** 2 + 1], color="#e53e3e", s=40, zorder=5)
        ax.scatter([steps_small[-1]], [(steps_small[-1] - 2) ** 2 + 1],
                   color="#38a169", s=120, marker="*", zorder=6, label="Convergência")
        ax.set_xlabel("β")
        ax.set_ylabel("J(β)")
        ax.set_title(f"η = {eta_small} — Passos Pequenos", fontweight="bold")
        ax.legend(fontsize=8)
        fig.tight_layout()
        save_and_show(fig, "gd_passos_pequenos")

    with col_gd2:
        st.markdown("**Passos grandes (η = 0.9) — oscilação / divergência**")
        eta_large = 0.9
        beta_t2 = -1.5
        steps_large = [beta_t2]
        for _ in range(10):
            grad = 2 * (beta_t2 - 2)
            beta_t2 = beta_t2 - eta_large * grad
            steps_large.append(beta_t2)
            if abs(beta_t2) > 20:
                break

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(beta_vals, J_vals, color="#667eea", linewidth=2.5, label="J(β) = (β-2)²+1")
        for s in steps_large:
            ax.scatter([s], [(s - 2) ** 2 + 1], color="#e53e3e", s=40, zorder=5)
        ax.set_xlabel("β")
        ax.set_ylabel("J(β)")
        ax.set_title(f"η = {eta_large} — Passos Grandes (oscila)", fontweight="bold")
        ax.legend(fontsize=8)
        fig.tight_layout()
        save_and_show(fig, "gd_passos_grandes")

    divider()

    caixa_conceito("""
    <b>Taxa de Aprendizado (Learning Rate η):</b><br>
    &nbsp;&nbsp;• η <b>muito pequena</b>: converge lentamente — pode precisar de muitas épocas<br>
    &nbsp;&nbsp;• η <b>muito grande</b>: oscila ou diverge — nunca encontra o mínimo<br>
    &nbsp;&nbsp;• η <b>adequada</b>: converge em poucas épocas com boa precisão<br><br>
    <b>Épocas:</b> número de vezes que o algoritmo percorre todo o dataset de treino.
    """)

    divider()
    secao("⚙️ SGDRegressor no Dataset Insurance")

    col_eta, col_iter = st.columns(2)
    with col_eta:
        eta0 = st.slider("Taxa de aprendizado η₀", min_value=0.001, max_value=0.1,
                         value=0.01, step=0.001, format="%.3f")
    with col_iter:
        max_iter = st.slider("Número máximo de iterações (épocas)", min_value=100,
                             max_value=2000, value=500, step=100)

    try:
        df_gd = carregar_insurance().dropna().drop_duplicates().reset_index(drop=True)

        num_f = ["age", "bmi", "children"]
        cat_f = ["sex", "smoker", "region"]

        X_gd = df_gd[num_f + cat_f]
        y_gd = df_gd["charges"].values

        X_tr_gd, X_te_gd, y_tr_gd, y_te_gd = train_test_split(
            X_gd, y_gd, test_size=0.2, random_state=42
        )

        pre_gd = ColumnTransformer(transformers=[
            ("num", StandardScaler(), num_f),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_f),
        ])

        # LinearRegression (solução analítica)
        pipe_lr = Pipeline([("pre", pre_gd), ("model", LinearRegression())])
        pipe_lr.fit(X_tr_gd, y_tr_gd)
        y_pred_lr_gd = pipe_lr.predict(X_te_gd)

        # SGDRegressor
        pre_gd2 = ColumnTransformer(transformers=[
            ("num", StandardScaler(), num_f),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_f),
        ])
        pipe_sgd = Pipeline([
            ("pre", pre_gd2),
            ("model", SGDRegressor(eta0=eta0, max_iter=max_iter,
                                   random_state=42, learning_rate="constant")),
        ])
        pipe_sgd.fit(X_tr_gd, y_tr_gd)
        y_pred_sgd = pipe_sgd.predict(X_te_gd)

        r2_lr_gd   = r2_score(y_te_gd, y_pred_lr_gd)
        rmse_lr_gd = np.sqrt(mean_squared_error(y_te_gd, y_pred_lr_gd))
        r2_sgd     = r2_score(y_te_gd, y_pred_sgd)
        rmse_sgd   = np.sqrt(mean_squared_error(y_te_gd, y_pred_sgd))

        st.markdown("**Comparação de Modelos:**")
        df_comp = pd.DataFrame({
            "Modelo": ["LinearRegression (analítico)", f"SGDRegressor (η={eta0}, iter={max_iter})"],
            "R²": [f"{r2_lr_gd:.4f}", f"{r2_sgd:.4f}"],
            "RMSE (USD)": [f"${rmse_lr_gd:,.0f}", f"${rmse_sgd:,.0f}"],
        })
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        col_pp1, col_pp2 = st.columns(2)
        with col_pp1:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(y_te_gd, y_pred_lr_gd, alpha=0.5, color="#667eea", s=25)
            lims = [0, max(y_te_gd.max(), y_pred_lr_gd.max())]
            ax.plot(lims, lims, "k--", linewidth=1.5)
            ax.set_xlabel("Real")
            ax.set_ylabel("Previsto")
            ax.set_title(f"LinearRegression (R²={r2_lr_gd:.3f})", fontweight="bold")
            fig.tight_layout()
            save_and_show(fig, "gd_paridade_lr")

        with col_pp2:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(y_te_gd, y_pred_sgd, alpha=0.5, color="#764ba2", s=25)
            ax.plot(lims, lims, "k--", linewidth=1.5)
            ax.set_xlabel("Real")
            ax.set_ylabel("Previsto")
            ax.set_title(f"SGDRegressor (R²={r2_sgd:.3f})", fontweight="bold")
            fig.tight_layout()
            save_and_show(fig, "gd_paridade_sgd")

    except Exception as e:
        caixa_atencao(f"Não foi possível carregar o dataset Insurance.<br><code>{e}</code>")

    divider()

    with st.expander("📝 Ver código desta seção"):
        code_block("""
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# SGDRegressor: usa Gradiente Descendente Estocástico
# eta0: taxa de aprendizado inicial
# max_iter: número de épocas (passagens pelo dataset)
# learning_rate='constant': mantém eta0 fixo durante todo o treino

pipe_sgd = Pipeline([
    ("preprocessor", preprocessor),    # mesma lógica que LinearRegression
    ("model", SGDRegressor(
        eta0=0.01,
        max_iter=500,
        random_state=42,
        learning_rate="constant",
    )),
])
pipe_sgd.fit(X_train, y_train)
y_pred = pipe_sgd.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"SGD — R²={r2:.4f}  RMSE={rmse:.2f}")
""")

    st.markdown("""
    <div class="footer">
        <p style="margin:0; font-size:1rem; font-weight:600;">
            Machine Learning — Aula 03 · Gradiente Descendente
        </p>
        <p style="margin:0.3rem 0 0; font-size:0.85rem; opacity:0.7;">
            Gráficos salvos em <code>outputs/</code>
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# PÁGINA: REGRESSÃO POLINOMIAL
# ============================================================================

elif pagina == "🔵  Regressão Polinomial":

    st.markdown("""
    <div style='text-align:center; padding: 1.5rem 0 0.5rem;'>
        <h1 style='font-size:2.4rem; font-weight:800; color:#1a1a2e;'>
            🔵 Regressão Polinomial
        </h1>
        <p style='font-size:1.05rem; color:#555;'>
            Modelar relações não-lineares com PolynomialFeatures
        </p>
    </div>
    """, unsafe_allow_html=True)

    divider()

    caixa_conceito("""
    <b>Regressão Polinomial</b> estende a regressão linear adicionando potências das features:<br><br>
    <b>Grau 2:</b> <code>ŷ = β₀ + β₁x + β₂x²</code><br>
    <b>Grau 3:</b> <code>ŷ = β₀ + β₁x + β₂x² + β₃x³</code><br>
    <b>Grau d:</b> <code>ŷ = β₀ + Σᵢ βᵢxⁱ</code><br><br>
    O modelo ainda é <b>linear nos parâmetros</b> — usamos
    <code>PolynomialFeatures</code> para criar as novas colunas e
    <code>LinearRegression</code> para ajustar os coeficientes.
    """)

    divider()

    tab_didatico, tab_ins_poly = st.tabs(["📚 Exemplo Didático", "🏥 Dataset Insurance"])

    # ------------------------------------------------------------------
    # TAB 1: EXEMPLO DIDÁTICO
    # ------------------------------------------------------------------
    with tab_didatico:

        secao("📊 Dados: y = 1 - exp(-X) + ruído")

        grau = st.slider("Grau do polinômio", min_value=1, max_value=8, value=2, step=1,
                         key="grau_didatico")

        np.random.seed(7)
        X_poly_raw = np.sort(np.random.uniform(0, 5, 120))
        y_poly_raw = 1 - np.exp(-X_poly_raw) + np.random.normal(0, 0.08, 120)

        X_poly = X_poly_raw.reshape(-1, 1)
        X_tr_p, X_te_p, y_tr_p, y_te_p = train_test_split(
            X_poly, y_poly_raw, test_size=0.2, random_state=42
        )

        # Regressão linear
        lin_p = LinearRegression()
        lin_p.fit(X_tr_p, y_tr_p)
        y_pred_lin_p = lin_p.predict(X_te_p)

        # Regressão polinomial
        poly_pipe = Pipeline([
            ("poly", PolynomialFeatures(degree=grau, include_bias=False)),
            ("model", LinearRegression()),
        ])
        poly_pipe.fit(X_tr_p, y_tr_p)
        y_pred_poly_p = poly_pipe.predict(X_te_p)

        r2_lin_p   = r2_score(y_te_p, y_pred_lin_p)
        rmse_lin_p = np.sqrt(mean_squared_error(y_te_p, y_pred_lin_p))
        mae_lin_p  = mean_absolute_error(y_te_p, y_pred_lin_p)

        r2_pol_p   = r2_score(y_te_p, y_pred_poly_p)
        rmse_pol_p = np.sqrt(mean_squared_error(y_te_p, y_pred_poly_p))
        mae_pol_p  = mean_absolute_error(y_te_p, y_pred_poly_p)

        # Gráfico
        x_curve = np.linspace(0, 5, 300).reshape(-1, 1)
        y_curve_lin  = lin_p.predict(x_curve)
        y_curve_poly = poly_pipe.predict(x_curve)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(X_poly_raw, y_poly_raw, alpha=0.4, color="#667eea",
                   s=20, label="Dados")
        ax.plot(x_curve, y_curve_lin, color="#e53e3e", linewidth=2,
                label=f"Linear (R²={r2_lin_p:.3f})", linestyle="--")
        ax.plot(x_curve, y_curve_poly, color="#38a169", linewidth=2,
                label=f"Polinomial grau {grau} (R²={r2_pol_p:.3f})")
        ax.set_xlabel("X")
        ax.set_ylabel("y")
        ax.set_title(f"Regressão: Linear vs Polinomial (grau {grau})", fontweight="bold")
        ax.legend(fontsize=9)
        fig.tight_layout()
        save_and_show(fig, f"poly_didatico_grau{grau}")

        divider()
        secao("📊 Comparação de Métricas")
        df_metricas = pd.DataFrame({
            "Modelo": [f"Linear (grau 1)", f"Polinomial (grau {grau})"],
            "R²":     [f"{r2_lin_p:.4f}", f"{r2_pol_p:.4f}"],
            "MAE":    [f"{mae_lin_p:.4f}", f"{mae_pol_p:.4f}"],
            "RMSE":   [f"{rmse_lin_p:.4f}", f"{rmse_pol_p:.4f}"],
        })
        st.dataframe(df_metricas, use_container_width=True, hide_index=True)

        if grau >= 6:
            caixa_atencao("""
            <b>Cuidado com overfitting!</b> Com grau muito alto, o modelo decora os dados
            de treino (curva passa por quase todos os pontos) mas generaliza mal para novos dados.
            Isso se manifesta como R² alto no treino, mas baixo no teste — ou oscilações
            violentas da curva fora dos dados observados.
            """)
        else:
            caixa_dica(f"""
            Grau {grau} está bem ajustado para esta distribuição.
            Compare o R² linear ({r2_lin_p:.3f}) com o polinomial ({r2_pol_p:.3f})
            para ver o ganho de expressividade do modelo.
            """)

        divider()
        secao("🔍 Expansão de features com PolynomialFeatures")

        caixa_conceito("""
        <b>Como PolynomialFeatures funciona:</b><br>
        Para uma feature X e grau 2, a transformação cria: <code>[X, X²]</code><br>
        Para duas features [X₁, X₂] e grau 2, cria: <code>[X₁, X₂, X₁², X₁X₂, X₂²]</code><br><br>
        O número de features cresce rapidamente com o grau e o número de variáveis originais.
        """)

        x_ex = np.array([[2.0], [3.0], [4.0]])
        st.markdown("**Exemplo: X = [[2], [3], [4]] com grau 3**")
        pf_ex = PolynomialFeatures(degree=3, include_bias=False)
        x_ex_transformed = pf_ex.fit_transform(x_ex)
        df_ex = pd.DataFrame(
            x_ex_transformed,
            columns=[f"X^{i+1}" for i in range(x_ex_transformed.shape[1])],
        )
        st.dataframe(df_ex, use_container_width=True)

    # ------------------------------------------------------------------
    # TAB 2: INSURANCE POLINOMIAL
    # ------------------------------------------------------------------
    with tab_ins_poly:

        secao("🏥 Regressão Polinomial — Dataset Insurance")

        grau_ins = st.slider("Grau do polinômio", min_value=1, max_value=5, value=2,
                             key="grau_insurance")

        try:
            df_ins_p = carregar_insurance().dropna().drop_duplicates().reset_index(drop=True)

            num_fp = ["age", "bmi", "children"]
            cat_fp = ["sex", "smoker", "region"]

            X_inp = df_ins_p[num_fp + cat_fp]
            y_inp = df_ins_p["charges"].values

            X_tr_ip, X_te_ip, y_tr_ip, y_te_ip = train_test_split(
                X_inp, y_inp, test_size=0.2, random_state=42
            )

            pre_ip = ColumnTransformer(transformers=[
                ("num", StandardScaler(), num_fp),
                ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_fp),
            ])

            # Grau 1 (baseline)
            pipe_ip1 = Pipeline([
                ("pre", ColumnTransformer(transformers=[
                    ("num", StandardScaler(), num_fp),
                    ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_fp),
                ])),
                ("model", LinearRegression()),
            ])
            pipe_ip1.fit(X_tr_ip, y_tr_ip)
            y_pred_ip1 = pipe_ip1.predict(X_te_ip)

            # Grau selecionado
            pipe_ipd = Pipeline([
                ("pre", pre_ip),
                ("poly", PolynomialFeatures(degree=grau_ins, include_bias=False)),
                ("model", LinearRegression()),
            ])
            pipe_ipd.fit(X_tr_ip, y_tr_ip)
            y_pred_ipd = pipe_ipd.predict(X_te_ip)

            r2_ip1   = r2_score(y_te_ip, y_pred_ip1)
            rmse_ip1 = np.sqrt(mean_squared_error(y_te_ip, y_pred_ip1))
            mae_ip1  = mean_absolute_error(y_te_ip, y_pred_ip1)

            r2_ipd   = r2_score(y_te_ip, y_pred_ipd)
            rmse_ipd = np.sqrt(mean_squared_error(y_te_ip, y_pred_ipd))
            mae_ipd  = mean_absolute_error(y_te_ip, y_pred_ipd)

            df_comp_ip = pd.DataFrame({
                "Modelo": ["Grau 1 (Linear)", f"Grau {grau_ins} (Polinomial)"],
                "R²":     [f"{r2_ip1:.4f}", f"{r2_ipd:.4f}"],
                "MAE":    [f"${mae_ip1:,.0f}", f"${mae_ipd:,.0f}"],
                "RMSE":   [f"${rmse_ip1:,.0f}", f"${rmse_ipd:,.0f}"],
            })
            st.dataframe(df_comp_ip, use_container_width=True, hide_index=True)

            secao("🎯 Gráfico de Paridade — Polinomial Insurance")
            col_ppi1, col_ppi2 = st.columns(2)
            with col_ppi1:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.scatter(y_te_ip, y_pred_ip1, alpha=0.5, color="#667eea", s=25)
                lims_ip = [0, max(y_te_ip.max(), y_pred_ip1.max())]
                ax.plot(lims_ip, lims_ip, "k--", linewidth=1.5)
                ax.set_xlabel("Real")
                ax.set_ylabel("Previsto")
                ax.set_title(f"Grau 1 (R²={r2_ip1:.3f})", fontweight="bold")
                fig.tight_layout()
                save_and_show(fig, "poly_insurance_grau1")

            with col_ppi2:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.scatter(y_te_ip, y_pred_ipd, alpha=0.5, color="#38a169", s=25)
                lims_ipd = [0, max(y_te_ip.max(), max(y_pred_ipd))]
                ax.plot(lims_ipd, lims_ipd, "k--", linewidth=1.5)
                ax.set_xlabel("Real")
                ax.set_ylabel("Previsto")
                ax.set_title(f"Grau {grau_ins} (R²={r2_ipd:.3f})", fontweight="bold")
                fig.tight_layout()
                save_and_show(fig, f"poly_insurance_grau{grau_ins}")

            with st.expander("📝 Ver código desta seção"):
                code_block("""
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Pipeline com PolynomialFeatures aplicado APÓS o pré-processamento
pipe_poly = Pipeline([
    ("preprocessor", preprocessor),           # StandardScaler + OneHotEncoder
    ("poly", PolynomialFeatures(degree=2,      # cria features polinomiais
                                include_bias=False)),
    ("model", LinearRegression()),
])
pipe_poly.fit(X_train, y_train)
y_pred = pipe_poly.predict(X_test)
""")

        except Exception as e:
            caixa_atencao(f"Não foi possível carregar o dataset Insurance.<br><code>{e}</code>")

    st.markdown("""
    <div class="footer">
        <p style="margin:0; font-size:1rem; font-weight:600;">
            Machine Learning — Aula 03 · Regressão Polinomial
        </p>
        <p style="margin:0.3rem 0 0; font-size:0.85rem; opacity:0.7;">
            Gráficos salvos em <code>outputs/</code>
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# PÁGINA: DATASET REAL — CALIFORNIA HOUSING
# ============================================================================

elif pagina == "🏘️  Dataset Real — Imóveis (California Housing)":

    st.markdown("""
    <div style='text-align:center; padding: 1.5rem 0 0.5rem;'>
        <h1 style='font-size:2.4rem; font-weight:800; color:#1a1a2e;'>
            🏘️ Dataset Real — California Housing
        </h1>
        <p style='font-size:1.05rem; color:#555;'>
            20640 imóveis da Califórnia — pipeline completo com scikit-learn
        </p>
    </div>
    """, unsafe_allow_html=True)

    divider()

    try:
        df_cal, feat_names, _ = carregar_california()
        target_col = "MedHouseVal"

        caixa_conceito(f"""
        <b>California Housing Dataset (sklearn):</b><br>
        &nbsp;&nbsp;• <b>Amostras:</b> {len(df_cal):,}<br>
        &nbsp;&nbsp;• <b>Features:</b> {len(feat_names)} ({', '.join(feat_names)})<br>
        &nbsp;&nbsp;• <b>Alvo:</b> MedHouseVal — valor mediano dos imóveis (em $100.000)<br><br>
        Dados coletados pelo censo americano de 1990. Cada linha representa
        um bloco censitário na Califórnia.
        """)

        st.dataframe(df_cal.head(), use_container_width=True)

        divider()

        X_cal = df_cal[list(feat_names)].values
        y_cal = df_cal[target_col].values

        X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
            X_cal, y_cal, test_size=0.2, random_state=42
        )

        tab_lin_cal, tab_poly_cal = st.tabs(["📐 Regressão Linear", "🔵 Regressão Polinomial"])

        # ------------------------------------------------------------------
        # TAB 1: LINEAR
        # ------------------------------------------------------------------
        with tab_lin_cal:

            secao("📐 Pipeline: StandardScaler + LinearRegression")

            pipe_cal_lin = Pipeline([
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ])
            pipe_cal_lin.fit(X_tr_c, y_tr_c)
            y_pred_cl = pipe_cal_lin.predict(X_te_c)

            r2_cl   = r2_score(y_te_c, y_pred_cl)
            rmse_cl = np.sqrt(mean_squared_error(y_te_c, y_pred_cl))
            mae_cl  = mean_absolute_error(y_te_c, y_pred_cl)

            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.markdown(f'<div class="metric-card"><h2>{r2_cl:.3f}</h2><p>R²</p></div>',
                            unsafe_allow_html=True)
            with mc2:
                st.markdown(f'<div class="metric-card"><h2>{mae_cl:.3f}</h2><p>MAE ($100k)</p></div>',
                            unsafe_allow_html=True)
            with mc3:
                st.markdown(f'<div class="metric-card"><h2>{rmse_cl:.3f}</h2><p>RMSE ($100k)</p></div>',
                            unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(y_te_c, y_pred_cl, alpha=0.3, color="#667eea", s=10)
            lims_c = [y_te_c.min(), y_te_c.max()]
            ax.plot(lims_c, lims_c, "k--", linewidth=1.5, label="Perfeito")
            ax.set_xlabel("MedHouseVal Real ($100k)")
            ax.set_ylabel("MedHouseVal Previsto ($100k)")
            ax.set_title(f"Paridade — California Housing Linear (R²={r2_cl:.3f})", fontweight="bold")
            ax.legend(fontsize=9)
            fig.tight_layout()
            save_and_show(fig, "california_paridade_linear")

            st.markdown(
                f"<div class='explain-box'><b>Interpretação do R² = {r2_cl:.3f}:</b><br>"
                f"O modelo linear explica {r2_cl*100:.1f}% da variância no valor dos imóveis. "
                f"Um R² de {r2_cl:.2f} é razoável para previsão de preços, indicando que as "
                f"8 features capturam boa parte da variabilidade, mas há padrões não-lineares "
                f"que o modelo linear não consegue capturar completamente.</div>",
                unsafe_allow_html=True,
            )

            with st.expander("📝 Ver código desta seção"):
                code_block("""
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

# Carregar o dataset California Housing
housing = fetch_california_housing(as_frame=True)
df = housing.frame

X = df[housing.feature_names].values
y = df["MedHouseVal"].values  # em $100.000

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline: padronizar features → regressão linear
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression()),
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
print(f"R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")
# Lembrete: unidade é $100.000, então RMSE=0.7 ≈ $70.000 de erro médio
""")

        # ------------------------------------------------------------------
        # TAB 2: POLINOMIAL
        # ------------------------------------------------------------------
        with tab_poly_cal:

            secao("🔵 Pipeline: StandardScaler + PolynomialFeatures + LinearRegression")

            grau_cal = st.slider("Grau do polinômio", min_value=1, max_value=3, value=2,
                                 key="grau_california")

            caixa_dica("""
            Para o California Housing usamos grau máximo 3 pois as 8 features
            geram muitas combinações com grau alto, tornando o treino lento e
            suscetível a overfitting. Com grau 2 já obtemos melhora significativa.
            """)

            pipe_cal_poly = Pipeline([
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=grau_cal, include_bias=False)),
                ("model", LinearRegression()),
            ])
            pipe_cal_poly.fit(X_tr_c, y_tr_c)
            y_pred_cp = pipe_cal_poly.predict(X_te_c)

            r2_cp   = r2_score(y_te_c, y_pred_cp)
            rmse_cp = np.sqrt(mean_squared_error(y_te_c, y_pred_cp))
            mae_cp  = mean_absolute_error(y_te_c, y_pred_cp)

            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.markdown(f'<div class="metric-card"><h2>{r2_cp:.3f}</h2><p>R²</p></div>',
                            unsafe_allow_html=True)
            with mc2:
                st.markdown(f'<div class="metric-card"><h2>{mae_cp:.3f}</h2><p>MAE ($100k)</p></div>',
                            unsafe_allow_html=True)
            with mc3:
                st.markdown(f'<div class="metric-card"><h2>{rmse_cp:.3f}</h2><p>RMSE ($100k)</p></div>',
                            unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(y_te_c, y_pred_cp, alpha=0.3, color="#38a169", s=10)
            ax.plot(lims_c, lims_c, "k--", linewidth=1.5, label="Perfeito")
            ax.set_xlabel("MedHouseVal Real ($100k)")
            ax.set_ylabel("MedHouseVal Previsto ($100k)")
            ax.set_title(f"Paridade — Polinomial grau {grau_cal} (R²={r2_cp:.3f})",
                         fontweight="bold")
            ax.legend(fontsize=9)
            fig.tight_layout()
            save_and_show(fig, f"california_paridade_poly_grau{grau_cal}")

            divider()
            secao("⚖️ Comparação: Linear vs Polinomial")

            df_comp_cal = pd.DataFrame({
                "Modelo": [
                    "Linear (grau 1)",
                    f"Polinomial (grau {grau_cal})",
                ],
                "R²":     [f"{r2_cl:.4f}", f"{r2_cp:.4f}"],
                "MAE":    [f"{mae_cl:.4f}", f"{mae_cp:.4f}"],
                "RMSE":   [f"{rmse_cl:.4f}", f"{rmse_cp:.4f}"],
            })
            st.dataframe(df_comp_cal, use_container_width=True, hide_index=True)

            melhora_r2 = (r2_cp - r2_cl) * 100
            st.markdown(
                f"<div class='explain-box'>"
                f"<b>Melhora com grau {grau_cal}:</b> R² aumentou "
                f"{melhora_r2:+.2f} pontos percentuais em relação ao modelo linear.</div>",
                unsafe_allow_html=True,
            )

            with st.expander("📝 Ver código desta seção"):
                code_block("""
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Para California Housing, recomenda-se grau 2 (bom custo-benefício)
# grau 3 pode ser lento devido ao grande número de features geradas
pipe_poly = Pipeline([
    ("scaler", StandardScaler()),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("model", LinearRegression()),
])
pipe_poly.fit(X_train, y_train)
y_pred = pipe_poly.predict(X_test)

r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"R²={r2:.4f}  RMSE={rmse:.4f}")
""")

    except Exception as e:
        caixa_atencao(f"Não foi possível carregar o dataset California Housing.<br><code>{e}</code>")

    st.markdown("""
    <div class="footer">
        <p style="margin:0; font-size:1rem; font-weight:600;">
            Machine Learning — Aula 03 · Dataset Real — California Housing
        </p>
        <p style="margin:0.3rem 0 0; font-size:0.85rem; opacity:0.7;">
            Gráficos salvos em <code>outputs/</code>
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# FOOTER GLOBAL — ASSINATURA
# ============================================================================

st.markdown("---")
st.markdown(
    "<div style='text-align:center; padding:1rem 0; color:#888; font-size:0.85rem;'>"
    "Cláudio Ferreira Neves &nbsp;·&nbsp; Especialista em Ciência de Dados e IA"
    "</div>",
    unsafe_allow_html=True,
)
