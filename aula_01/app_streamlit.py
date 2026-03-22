"""
=============================================================================
Machine Learning — Aula 01
Introdução ao Machine Learning
Aplicação Streamlit interativa — material didático para iniciantes

Autor       : Cláudio Ferreira Neves
Cargo atual : Analista de BI — Save Co. | Jaraguá do Sul/SC
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

import seaborn as sns
import streamlit as st
from sklearn import datasets
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
)
import pandas as pd

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


def plot_decision_boundary(model, X: np.ndarray, y: np.ndarray, title: str = "") -> plt.Figure:
    """
    Gera o gráfico da fronteira de decisão de um classificador.

    A ideia é: criamos uma grade densa de pontos cobrindo todo o espaço 2D,
    perguntamos ao modelo qual classe ele prevê para cada ponto da grade,
    e colorimos as regiões de acordo. Assim enxergamos visualmente onde o
    modelo 'traça a linha' entre as classes.
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.contourf(xx, yy, Z, cmap="bwr", alpha=0.25)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k", s=55, zorder=3)
    ax.set_title(title or "Fronteira de Decisão", fontsize=13, fontweight="bold")
    ax.set_xlabel("Feature 1 (característica 1)")
    ax.set_ylabel("Feature 2 (característica 2)")
    fig.tight_layout()
    return fig


# ============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="Introdução ao ML",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS — visual premium
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
        <h2 style="margin:0; font-size:1.3rem;">🤖 Machine Learning</h2>
        <p style="margin:0.4rem 0 0; font-size:0.8rem; opacity:0.8;">Introdução Prática</p>
    </div>
    """, unsafe_allow_html=True)

    pagina = st.radio(
        "Navegação",
        options=[
            "🏠  Início",
            "📊  Representação de Dados",
            "📈  Regressão Linear Simples",
            "🔢  Regressão Polinomial",
            "🎯  Classificação Binária",
            "✂️   Divisão Treino / Teste",
            "🌸  Dataset IRIS — Pipeline Completo",
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
tab_nav(1)  # replace CURRENT_NUMBER with the correct int

st.markdown(
    "<p style='text-align:center; font-size:0.95rem; color:#667eea; "
    "font-weight:600; margin-bottom:0;'>"
    "Cláudio Ferreira Neves &nbsp;·&nbsp; Especialista em Ciência de Dados e IA"
    "</p>",
    unsafe_allow_html=True,
)

_nav_l, _nav_spacer, _nav_r = st.columns([1.2, 4, 1.2])
with _nav_l:
    if st.button("🏠 Portal", use_container_width=True, key="nav_portal"):
        st.switch_page(PAGE_PORTAL)
with _nav_r:
    if st.button("Aula 02 →", use_container_width=True, key="nav_next"):
        st.switch_page(PAGE_AULA_02)

# ============================================================================
# PÁGINA: INÍCIO
# ============================================================================

if pagina == "🏠  Início":

    st.markdown("""
    <div style='text-align:center; padding: 2rem 0 1rem;'>
        <h1 style='font-size:2.8rem; font-weight:800; color:#1a1a2e; margin-bottom:0.3rem;'>
            🤖 Introdução ao Machine Learning
        </h1>
        <p style='font-size:1.15rem; color:#555; max-width:700px; margin:0 auto;'>
            Aprenda os fundamentos de ML com exemplos interativos,
            visualizações e código explicado passo a passo.
        </p>
    </div>
    """, unsafe_allow_html=True)

    divider()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><h2>6</h2><p>Seções</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h2>3</h2><p>Algoritmos</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h2>📈</h2><p>Regressão</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><h2>🎯</h2><p>Classificação</p></div>', unsafe_allow_html=True)

    divider()

    st.markdown("## 📚 O que você vai aprender")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <div class="concept-box">
            <b>📊 Representação de Dados</b><br>
            Como estruturar dados em arrays NumPy e visualizá-los em gráficos de dispersão.
        </div>
        <div class="concept-box">
            <b>📈 Regressão Linear Simples</b><br>
            Modelar a relação linear entre variáveis. Solução analítica vs scikit-learn.
        </div>
        <div class="concept-box">
            <b>🔢 Regressão Polinomial</b><br>
            Capturar relações não-lineares com pipelines do scikit-learn.
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="concept-box">
            <b>🎯 Classificação Binária</b><br>
            Separar dados em classes usando Regressão Logística com fronteira de decisão.
        </div>
        <div class="concept-box">
            <b>✂️ Divisão Treino / Teste</b><br>
            Por que e como separar dados para avaliar modelos corretamente.
        </div>
        <div class="concept-box">
            <b>🌸 Pipeline IRIS Completo</b><br>
            Do zero à avaliação: carga → visualização → treino → métricas profissionais.
        </div>
        """, unsafe_allow_html=True)

    divider()
    st.markdown("## 🧰 Tecnologias utilizadas")
    cols = st.columns(5)
    for col, lib in zip(cols, ["NumPy", "Matplotlib", "Seaborn", "scikit-learn", "Streamlit"]):
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
# PÁGINA: REPRESENTAÇÃO DE DADOS
# ============================================================================

elif pagina == "📊  Representação de Dados":

    st.markdown('<p class="section-title">📊 Representação de Dados no Plano</p>', unsafe_allow_html=True)

    st.markdown("""
    Antes de treinar qualquer modelo de Machine Learning, precisamos entender
    como os dados são **armazenados** e **visualizados**. Tudo começa com a
    biblioteca **NumPy**, que trabalha com arrays (tabelas numéricas), e o
    **Matplotlib**, que transforma esses dados em gráficos.
    """)

    caixa_conceito("""
    <b>O que é um array NumPy?</b><br><br>
    Pense em uma planilha do Excel. Uma tabela com linhas e colunas — isso é
    um array 2D. No NumPy, chamamos esse formato de <code>(n_linhas, n_colunas)</code>.<br><br>
    Exemplo: <code>np.random.rand(20, 2)</code> cria uma tabela com 20 linhas e 2 colunas,
    onde cada linha representa um ponto no espaço 2D (com coordenada X e coordenada Y).
    """)

    divider()

    # --- Parâmetros interativos ---
    secao("⚙️ Ajuste os parâmetros abaixo e observe como o gráfico muda:")
    col1, col2, col3 = st.columns(3)
    with col1:
        n_pontos  = st.slider("Número de pontos", 10, 200, 20, step=10,
                              help="Quantas amostras (linhas) vão existir no array")
    with col2:
        n_classes = st.slider("Número de classes", 2, 6, 3,
                              help="Quantas categorias diferentes os pontos terão")
    with col3:
        seed      = int(st.number_input("Semente aleatória", value=42, step=1,
                              help="Garante que os mesmos números sejam gerados toda vez"))

    np.random.seed(seed)
    points = np.random.rand(n_pontos, 2) * 10
    labels = np.random.randint(0, n_classes, n_pontos)

    divider()

    # --- Indexação ---
    secao("🔍 Como acessar partes de um array?")
    st.markdown("""
    Imagine que `points` é uma tabela com 2 colunas: a **coluna 0** é o eixo X
    e a **coluna 1** é o eixo Y. Existem **três formas** de pegar a coluna X —
    e cada uma devolve um formato diferente:
    """)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        caixa_conceito("""
        <b><code>points[:, 0]</code></b><br><br>
        Pega a coluna 0 de <b>todas as linhas</b> (<code>:</code> = todas).<br>
        Resultado: vetor <b>1D</b> — uma lista simples de números.<br><br>
        <b>Problema:</b> o scikit-learn exige que X seja 2D. Este formato
        causaria erro!<br><br>
        Shape: <code>{}</code>
        """.format(points[:, 0].shape))

    with col_b:
        caixa_conceito("""
        <b><code>points[:, [0]]</code></b><br><br>
        O <code>[0]</code> entre colchetes diz ao NumPy: "quero uma lista
        com a coluna 0" — isso preserva a dimensão de colunas.<br><br>
        Resultado: matriz <b>2D</b> com 1 coluna.<br><br>
        Shape: <code>{}</code>
        """.format(points[:, [0]].shape))

    with col_c:
        caixa_conceito("""
        <b><code>points[:,0].reshape(-1, 1)</code></b><br><br>
        Pega o vetor 1D e o <b>reformata</b> para 2D.<br>
        O <code>-1</code> diz: "calcule o número de linhas automaticamente".<br><br>
        Resultado: matriz <b>2D</b> com 1 coluna.<br><br>
        Shape: <code>{}</code>
        """.format(points[:, 0].reshape(-1, 1).shape))

    caixa_dica("""
    <b>Regra de ouro:</b> sempre que for usar uma única feature no scikit-learn,
    use <code>reshape(-1, 1)</code>. O sklearn espera X com shape
    <code>(n_amostras, n_features)</code> — ou seja, sempre 2D.
    """)

    divider()

    # --- Visualizações ---
    secao("📉 Gráfico de Dispersão (Scatter Plot)")
    st.markdown("""
    O **gráfico de dispersão** é o mais usado em ML para explorar dados.
    Cada ponto no gráfico representa **uma amostra** — sua posição horizontal (X)
    é a primeira feature e a posição vertical (Y) é a segunda.
    """)

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(points[:, 0], points[:, 1], c="royalblue", alpha=0.75, edgecolors="k", s=70)
        ax.set_title("Scatter Plot — Cor única", fontweight="bold", fontsize=12)
        ax.set_xlabel("Feature 1 (eixo X) →  valores de 0 a 10")
        ax.set_ylabel("Feature 2 (eixo Y) →  valores de 0 a 10")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        save_and_show(fig, "01_scatter_sem_classes")
        st.markdown("""
        <div class="explain-box">
            🟦 Cada ponto azul é uma <b>amostra</b> (uma linha do array).<br>
            A posição X vem de <code>points[:,0]</code> e a posição Y de <code>points[:,1]</code>.<br>
            O parâmetro <code>alpha=0.75</code> deixa os pontos levemente transparentes
            para que sobreposições fiquem visíveis.
        </div>
        """, unsafe_allow_html=True)

    with col_g2:
        fig, ax = plt.subplots(figsize=(5, 4))
        sc = ax.scatter(points[:, 0], points[:, 1], c=labels, cmap="Set1",
                        alpha=0.75, edgecolors="k", s=70)
        ax.set_title("Scatter Plot — Por classe (cmap)", fontweight="bold", fontsize=12)
        ax.set_xlabel("Feature 1 (eixo X) →  valores de 0 a 10")
        ax.set_ylabel("Feature 2 (eixo Y) →  valores de 0 a 10")
        plt.colorbar(sc, ax=ax, label="Classe (0, 1, 2...)")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        save_and_show(fig, "02_scatter_com_classes")
        st.markdown("""
        <div class="explain-box">
            🎨 O parâmetro <code>c=labels</code> colore cada ponto de acordo com sua classe.<br>
            O <code>cmap="Set1"</code> define a paleta de cores — aqui cada número
            (0, 1, 2...) vira uma cor diferente.<br>
            A barra lateral (colorbar) mostra a correspondência cor → classe.
        </div>
        """, unsafe_allow_html=True)

    divider()

    with st.expander("📋 Código completo explicado linha a linha"):
        code_block("""
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Por que importar numpy?
# NumPy é a base de todo ML em Python. Ele fornece arrays multidimensionais
# muito mais rápidos que listas Python e funções matemáticas vetorizadas.
# ─────────────────────────────────────────────────────────────────────────────

# np.random.seed(42): fixa a semente do gerador de números aleatórios.
# O número 42 é arbitrário — o importante é ser SEMPRE o mesmo número.
# Isso garante REPRODUTIBILIDADE: qualquer pessoa que rodar este código
# obterá exatamente os mesmos dados.
np.random.seed(42)

n = 20

# np.random.rand(n, 2): cria um array de shape (20, 2) com valores entre 0 e 1.
# Multiplicar por 10 escala os valores para o intervalo [0, 10].
# Cada linha = 1 ponto; coluna 0 = coordenada X; coluna 1 = coordenada Y.
points = np.random.rand(n, 2) * 10

# np.random.randint(low=0, high=3, size=n): sorteia n inteiros em [0, 3).
# Esses números representam as CLASSES (rótulos) de cada ponto.
labels = np.random.randint(0, 3, n)

# ─────────────────────────────────────────────────────────────────────────────
# Gráfico de dispersão — sem distinção de classes
# ─────────────────────────────────────────────────────────────────────────────
plt.figure(figsize=(5, 5))   # tamanho da figura em polegadas (largura, altura)

plt.scatter(
    points[:, 0],   # eixo X: primeira coluna de todos os pontos
    points[:, 1],   # eixo Y: segunda coluna de todos os pontos
    c="blue",       # cor única para todos os pontos
    alpha=0.7       # transparência: 0=invisível, 1=opaco
)
plt.title("Pontos no Plano")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# Gráfico de dispersão — com colormap por classe
# Colormap (cmap) mapeia automaticamente cada valor numérico a uma cor.
# Referência: https://matplotlib.org/stable/users/explain/colors/colormaps.html
# ─────────────────────────────────────────────────────────────────────────────
plt.scatter(
    points[:, 0],
    points[:, 1],
    c=labels,       # cada ponto recebe a cor correspondente ao seu rótulo
    cmap="Set1",    # paleta de cores: Set1 tem cores bem distintas (ótimo para classes)
    alpha=0.7
)
plt.colorbar(label="Classe")
plt.show()
""")


# ============================================================================
# PÁGINA: REGRESSÃO LINEAR SIMPLES
# ============================================================================

elif pagina == "📈  Regressão Linear Simples":

    st.markdown('<p class="section-title">📈 Regressão Linear Simples</p>', unsafe_allow_html=True)

    caixa_conceito("""
    <b>O que é Regressão?</b><br><br>
    Regressão é o problema de <b>prever um número</b> — como o preço de uma casa,
    a temperatura de amanhã ou o consumo de combustível de um carro.<br><br>
    A <b>Regressão Linear Simples</b> assume que a relação entre a variável de entrada
    (X) e a saída (y) pode ser descrita por uma <b>reta</b>:
    <br><br>
    <code>ŷ = β₀ + β₁ × X</code><br><br>
    onde <b>β₀</b> é o intercepto (onde a reta corta o eixo Y) e <b>β₁</b>
    é a inclinação (o quanto y sobe quando X aumenta em 1 unidade).
    """)

    divider()

    secao("⚙️ Ajuste os parâmetros e veja como a reta muda:")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        samples = st.slider("Amostras", 20, 300, 100, step=10,
                            help="Mais amostras = modelo mais estável")
    with col2:
        true_b0 = st.slider("β₀ real (intercepto)", 0.0, 10.0, 4.0, step=0.5,
                            help="Valor verdadeiro do intercepto nos dados")
    with col3:
        true_b1 = st.slider("β₁ real (inclinação)", 0.0, 10.0, 3.0, step=0.5,
                            help="Valor verdadeiro da inclinação nos dados")
    with col4:
        noise   = st.slider("Ruído (σ)", 0.0, 5.0, 1.0, step=0.1,
                            help="Variação aleatória em torno da reta real")

    np.random.seed(42)
    X = 2 * np.random.rand(samples, 1)
    e = np.random.randn(samples, 1) * noise
    y = true_b0 + true_b1 * X + e

    # Cálculo manual
    mean_x    = np.mean(X)
    mean_y    = np.mean(y)
    numer     = np.sum((X - mean_x) * (y - mean_y))
    denom     = np.sum((X - mean_x) ** 2)
    b1_man    = numer / denom
    b0_man    = mean_y - b1_man * mean_x
    y_pred_m  = b0_man + b1_man * X

    # sklearn
    model     = LinearRegression()
    model.fit(X, y)
    y_pred    = model.predict(X)
    b0_sk     = model.intercept_[0]
    b1_sk     = model.coef_[0][0]
    r2        = model.score(X, y)
    rmse      = np.sqrt(mean_squared_error(y, y_pred))

    divider()

    # --- Métricas ---
    secao("📊 O que o modelo encontrou?")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("β₀ encontrado", f"{b0_sk:.3f}", delta=f"real: {true_b0:.1f}")
    col_m2.metric("β₁ encontrado", f"{b1_sk:.3f}", delta=f"real: {true_b1:.1f}")
    col_m3.metric("R² (qualidade)", f"{r2:.4f}")
    col_m4.metric("RMSE (erro médio)", f"{rmse:.4f}")

    caixa_dica("""
    O modelo encontrou valores <b>próximos mas não iguais</b> aos reais — isso é normal!
    O ruído nos dados impede o ajuste perfeito. Reduza o ruído no slider para ver
    β₀ e β₁ se aproximarem dos valores reais.
    """)

    divider()

    # --- Passo a passo da solução manual ---
    secao("🧮 Passo a passo: Solução Analítica (Mínimos Quadrados)")

    st.markdown("""
    A solução analítica calcula os coeficientes **diretamente** a partir de fórmulas
    matemáticas, sem precisar "treinar" nada. É chamada de **Mínimos Quadrados (OLS)**
    porque minimiza a soma dos quadrados dos erros entre os valores reais e previstos.
    """)

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        caixa_conceito(f"""
        <b>Passo 1 — Calcular as médias:</b><br>
        <code>mean_x = média de todos os X = {mean_x:.4f}</code><br>
        <code>mean_y = média de todos os y = {mean_y:.4f}</code><br><br>
        As médias são o "centro de gravidade" dos dados.
        """)
        caixa_conceito(f"""
        <b>Passo 2 — Calcular β₁ (inclinação):</b><br>
        <code>β₁ = Σ[(Xᵢ - X̄)(yᵢ - ȳ)] / Σ[(Xᵢ - X̄)²]</code><br><br>
        Numerador = como X e y variam <b>juntos</b> (covariância)<br>
        Denominador = como X varia <b>sozinho</b> (variância)<br><br>
        Resultado: β₁ = <b>{b1_man:.4f}</b>
        """)

    with col_p2:
        caixa_conceito(f"""
        <b>Passo 3 — Calcular β₀ (intercepto):</b><br>
        <code>β₀ = ȳ - β₁ × X̄</code><br><br>
        Garante que a reta <b>passe pelo ponto médio</b> dos dados.<br><br>
        Resultado: β₀ = <b>{b0_man:.4f}</b>
        """)
        caixa_conceito(f"""
        <b>Passo 4 — Previsão:</b><br>
        <code>ŷ = β₀ + β₁ × X</code><br><br>
        Aplicamos a fórmula da reta para calcular a previsão
        de cada ponto de dados.<br><br>
        Equação encontrada: <b>ŷ = {b0_man:.2f} + {b1_man:.2f} × X</b>
        """)

    divider()

    # --- Gráficos ---
    secao("📉 Visualizações")
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("**Dados reais + reta calculada manualmente**")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(X, y, color="steelblue", alpha=0.6, s=45, label="Dados reais", zorder=3)
        ax.plot(X, y_pred_m, color="green", lw=2.5, label=f"Reta manual: ŷ={b0_man:.2f}+{b1_man:.2f}X")
        ax.set_title("Solução Analítica (Mínimos Quadrados)", fontweight="bold")
        ax.set_xlabel("X → variável de entrada (independente)")
        ax.set_ylabel("y → variável de saída (dependente)")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        save_and_show(fig, "03_regressao_manual")
        st.markdown("""
        <div class="explain-box">
            🟦 <b>Pontos azuis</b>: dados reais com ruído.<br>
            🟢 <b>Reta verde</b>: melhor ajuste calculado pelas fórmulas matemáticas.<br>
            Perceba que a reta não passa por todos os pontos — ela minimiza
            a <b>soma dos quadrados das distâncias verticais</b> entre cada ponto e a reta.
        </div>
        """, unsafe_allow_html=True)

    with col_g2:
        st.markdown("**Manual vs scikit-learn (compare as retas)**")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(X, y, color="steelblue", alpha=0.6, s=45, label="Dados reais", zorder=3)
        ax.plot(X, y_pred_m, color="green",  lw=2.5, linestyle="--", label="Manual (fórmula)")
        ax.plot(X, y_pred,   color="crimson", lw=2,   label="scikit-learn (.fit)")
        ax.set_title("Manual vs scikit-learn — Resultados idênticos", fontweight="bold")
        ax.set_xlabel("X → variável de entrada")
        ax.set_ylabel("y → variável de saída")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        save_and_show(fig, "04_regressao_comparacao")
        st.markdown("""
        <div class="explain-box">
            🟢 <b>Tracejado verde</b>: solução pela fórmula matemática.<br>
            🔴 <b>Linha vermelha</b>: solução pelo scikit-learn.<br>
            As duas <b>coincidem completamente</b> — o sklearn implementa
            internamente o mesmo algoritmo OLS, mas de forma otimizada
            e pronta para dados com múltiplas features.
        </div>
        """, unsafe_allow_html=True)

    divider()

    secao("🔮 Faça uma previsão para um novo valor de X")
    st.markdown("""
    Agora que o modelo está treinado, podemos usá-lo para prever qualquer valor novo —
    mesmo valores que o modelo **nunca viu** durante o treino.
    """)

    novo_x_val = st.slider("Valor de X para prever:", float(X.min()), float(X.max()), 1.3, step=0.05)
    novo_x     = np.array([[novo_x_val]])
    novo_y     = model.predict(novo_x)[0][0]

    caixa_conceito(f"""
    Para <b>X = {novo_x_val:.2f}</b>, o modelo prevê <b>ŷ = {novo_y:.4f}</b><br><br>
    Cálculo: ŷ = {b0_sk:.4f} + {b1_sk:.4f} × {novo_x_val:.2f} = <b>{novo_y:.4f}</b>
    """)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(X, y, color="steelblue", alpha=0.5, s=40, label="Dados de treino", zorder=3)
    ax.plot(X, y_pred, color="crimson", lw=2, label="Modelo treinado")
    # Linhas tracejadas até o ponto previsto
    ax.axvline(novo_x_val, color="gray", linestyle=":", linewidth=1.2)
    ax.axhline(novo_y,     color="gray", linestyle=":", linewidth=1.2)
    ax.scatter(novo_x_val, novo_y, color="gold", edgecolors="k", s=250, marker="*",
               zorder=10, label=f"Previsão: x={novo_x_val:.2f} → ŷ={novo_y:.2f}")
    ax.set_title("Regressão Linear — Previsão para Novo Ponto", fontweight="bold")
    ax.set_xlabel("X → variável de entrada")
    ax.set_ylabel("y → variável de saída / previsão")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    save_and_show(fig, "05_regressao_previsao")
    st.markdown("""
    <div class="explain-box">
        ⭐ <b>Estrela amarela</b>: o ponto que queremos prever.<br>
        As <b>linhas tracejadas cinzas</b> mostram o caminho: sobe pelo eixo X
        até tocar a reta, depois vai horizontal até o eixo Y.<br>
        O valor onde a linha horizontal toca o eixo Y é a nossa <b>previsão</b>.
    </div>
    """, unsafe_allow_html=True)

    divider()

    secao("📐 Entendendo as Métricas de Avaliação")
    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        caixa_conceito(f"""
        <b>R² — Coeficiente de Determinação</b><br><br>
        Responde: <i>"Quanto da variação de y o meu modelo consegue explicar?"</i><br><br>
        <b>Valor atual: {r2:.4f}</b><br><br>
        • R² = <b>1.0</b> → ajuste perfeito (reta passa por todos os pontos)<br>
        • R² = <b>0.0</b> → o modelo não explica nada (pior que usar a média)<br>
        • R² = <b>0.9</b> → o modelo explica 90% da variação dos dados<br><br>
        <b>Analogia:</b> imagine que y é a nota de um aluno. Se R²=0.85, significa
        que 85% da variação nas notas é explicada pelo tempo de estudo (X).
        """)

    with col_exp2:
        caixa_conceito(f"""
        <b>RMSE — Raiz do Erro Quadrático Médio</b><br><br>
        Responde: <i>"Em média, o quanto minha previsão erra, em unidades de y?"</i><br><br>
        <b>Valor atual: {rmse:.4f}</b><br><br>
        • RMSE = <b>0</b> → sem erro (impossível com dados reais)<br>
        • Quanto maior o RMSE, mais longe as previsões ficam dos valores reais<br>
        • RMSE está na <b>mesma unidade</b> que y (ex: se y é em R$, RMSE é em R$)<br><br>
        <b>Por que elevar ao quadrado?</b> Para penalizar erros grandes mais do
        que erros pequenos e eliminar o sinal (positivo/negativo).
        """)

    caixa_atencao("""
    <b>R² alto não significa modelo perfeito!</b> Um R² de 0.99 no treino pode
    indicar que o modelo <b>memorizou</b> os dados em vez de aprender o padrão real
    (overfitting). Por isso precisamos sempre avaliar em dados de <b>teste</b> — veja a
    seção "Divisão Treino / Teste".
    """)

    divider()

    with st.expander("📋 Código completo explicado linha a linha"):
        code_block("""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ─────────────────────────────────────────────────────────────────────────────
# Por que importar LinearRegression do sklearn?
# O scikit-learn (sklearn) é a biblioteca padrão de ML em Python.
# Ela implementa dezenas de algoritmos testados, otimizados e com interface
# padronizada: .fit() para treinar, .predict() para prever.
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)
samples = 100

# Criação de dados artificiais seguindo a equação y = 4 + 3*X + ruído
# Isso simula uma relação linear real com variação natural dos dados.
X = 2 * np.random.rand(samples, 1)   # X ∈ [0, 2), shape (100, 1)
e = np.random.randn(samples, 1)       # ruído gaussiano ε ~ N(0,1)
y = 4 + 3 * X + e                    # equação verdadeira: β₀=4, β₁=3

# ─────────────────────────────────────────────────────────────────────────────
# SOLUÇÃO MANUAL: Fórmula dos Mínimos Quadrados Ordinários (OLS)
# ─────────────────────────────────────────────────────────────────────────────
mean_x = np.mean(X)   # média de X
mean_y = np.mean(y)   # média de y

# Numerador: covariância entre X e y (como eles variam juntos)
numer = np.sum((X - mean_x) * (y - mean_y))
# Denominador: variância de X (como X varia sozinho)
denom = np.sum((X - mean_x) ** 2)

b1 = numer / denom              # inclinação
b0 = mean_y - b1 * mean_x      # intercepto (garante que a reta passe pela média)
y_pred_manual = b0 + b1 * X    # aplicando a reta para todos os pontos

# ─────────────────────────────────────────────────────────────────────────────
# SOLUÇÃO COM SKLEARN: interface padronizada
#
# model = LinearRegression()  → cria o objeto do modelo (ainda vazio)
# model.fit(X, y)             → TREINA o modelo: encontra β₀ e β₁ ótimos
# model.predict(X)            → APLICA o modelo: calcula ŷ = β₀ + β₁X
#
# Por que usar sklearn em vez da fórmula manual?
# 1. Funciona para múltiplas features (X com várias colunas) — a fórmula manual não
# 2. Mais rápido para datasets grandes
# 3. Interface padrão com todo o ecossistema de ML do sklearn
# ─────────────────────────────────────────────────────────────────────────────
model = LinearRegression()
model.fit(X, y)                        # treina

print(f"β₀ (intercepto): {model.intercept_[0]:.4f}")  # [0] porque retorna array
print(f"β₁ (inclinação): {model.coef_[0][0]:.4f}")    # [0][0]: 1ª feature, 1ª saída

y_pred = model.predict(X)

# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS
# model.score(X, y) → calcula R² diretamente
# mean_squared_error → MSE; np.sqrt → converte para RMSE (mesma unidade de y)
# ─────────────────────────────────────────────────────────────────────────────
r2   = model.score(X, y)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print(f"R²:   {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# PREVISÃO PARA NOVO PONTO
# O modelo foi treinado com X ∈ [0, 2). Podemos prever qualquer X novo,
# mesmo que ele não tenha estado nos dados de treino.
# np.array([[1.3]]): shape (1, 1) — 1 amostra, 1 feature
# ─────────────────────────────────────────────────────────────────────────────
novo_x = np.array([[1.3]])
novo_y = model.predict(novo_x)
print(f"Para X=1.3 → ŷ={novo_y[0][0]:.4f}")
""")


# ============================================================================
# PÁGINA: REGRESSÃO POLINOMIAL
# ============================================================================

elif pagina == "🔢  Regressão Polinomial":

    st.markdown('<p class="section-title">🔢 Regressão Polinomial</p>', unsafe_allow_html=True)

    caixa_conceito("""
    <b>Quando a reta não é suficiente...</b><br><br>
    A regressão linear assume que a relação entre X e y é uma reta.
    Mas e quando os dados formam uma <b>curva</b>?<br><br>
    A <b>Regressão Polinomial</b> resolve isso criando novas variáveis —
    potências de X — e ajustando uma reta sobre elas:<br><br>
    <code>ŷ = β₀ + β₁X + β₂X² + β₃X³ + ... + βdXᵈ</code><br><br>
    O modelo ainda é <b>linear nos parâmetros β</b> (por isso usamos LinearRegression
    internamente), mas a curva resultante pode ter qualquer forma.
    """)

    divider()

    secao("⚙️ Ajuste os parâmetros:")
    col1, col2, col3 = st.columns(3)
    with col1:
        samples_p = st.slider("Amostras", 30, 300, 100, step=10)
    with col2:
        degree    = st.slider("Grau do polinômio (d)", 1, 12, 2,
                              help="Grau 1 = reta; Grau 2 = parábola; graus altos = curvas complexas")
    with col3:
        noise_p   = st.slider("Ruído (σ)", 0.0, 0.5, 0.05, step=0.01)

    np.random.seed(42)
    X_p = np.random.rand(samples_p, 1) * 10
    e_p = np.random.randn(samples_p, 1) * noise_p
    y_p = 1 - np.exp(-X_p) + e_p   # curva real: y = 1 - e^(-x)

    modelo_pol = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    modelo_pol.fit(X_p, y_p)

    X_grid  = np.linspace(X_p.min(), X_p.max(), 400).reshape(-1, 1)
    y_grid  = modelo_pol.predict(X_grid)
    y_pred_p = modelo_pol.predict(X_p)
    r2_p    = modelo_pol.score(X_p, y_p)
    rmse_p  = np.sqrt(mean_squared_error(y_p, y_pred_p))

    divider()

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Grau do Polinômio", degree)
    col_m2.metric("R²", f"{r2_p:.4f}")
    col_m3.metric("RMSE", f"{rmse_p:.4f}")

    divider()

    secao("🔧 Como funciona o Pipeline?")
    st.markdown("""
    O scikit-learn tem um conceito poderoso chamado **Pipeline**: encadeia
    várias etapas de transformação + modelo de forma organizada e segura.
    """)

    col_pipe1, col_pipe2, col_pipe3 = st.columns(3)
    with col_pipe1:
        caixa_conceito(f"""
        <b>Etapa 1: PolynomialFeatures(degree={degree})</b><br><br>
        Recebe X e cria novas colunas: X², X³, ..., Xᵈ<br><br>
        Exemplo com degree=2:<br>
        <code>X = [3]  →  [1, 3, 9]</code><br>
        (1 = constante, 3 = X, 9 = X²)<br><br>
        O modelo passa a ter <b>{degree+1} features</b> em vez de 1.
        """)
    with col_pipe2:
        caixa_conceito(f"""
        <b>Etapa 2: LinearRegression()</b><br><br>
        Recebe as <b>{degree+1} colunas</b> e ajusta uma reta no espaço
        multidimensional.<br><br>
        Encontra β₀, β₁, β₂, ..., β{degree} que minimizam o erro.<br><br>
        A curva resultante no espaço original é um polinômio de grau {degree}.
        """)
    with col_pipe3:
        caixa_conceito(f"""
        <b>Por que usar Pipeline?</b><br><br>
        O Pipeline aplica automaticamente a <b>mesma transformação</b>
        tanto nos dados de treino quanto nos dados de teste.<br><br>
        Se fizéssemos manual, poderíamos errar e aplicar transformações
        diferentes — o Pipeline elimina esse risco.<br><br>
        É a forma <b>profissional</b> e segura de encadear etapas em ML.
        """)

    divider()

    secao("📉 Ajuste do Modelo Atual")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(X_p, y_p, color="steelblue", alpha=0.6, s=40, label="Dados reais", zorder=3)
    ax.plot(X_grid, y_grid, color="crimson", lw=2.5, label=f"Polinômio grau {degree}")
    # linha da função real
    y_real = 1 - np.exp(-X_grid)
    ax.plot(X_grid, y_real, color="green", lw=1.5, linestyle="--", alpha=0.7, label="Função real: y = 1 − e⁻ˣ")
    ax.set_title(f"Regressão Polinomial — Grau {degree}  |  R² = {r2_p:.4f}", fontweight="bold")
    ax.set_xlabel("X → variável de entrada")
    ax.set_ylabel("y → variável de saída")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    save_and_show(fig, f"06_polinomial_grau{degree}")
    st.markdown(f"""
    <div class="explain-box">
        🟦 <b>Pontos azuis</b>: dados reais com ruído.<br>
        🔴 <b>Curva vermelha</b>: polinômio de grau {degree} ajustado pelo modelo.<br>
        🟢 <b>Tracejado verde</b>: a função verdadeira que gerou os dados (y = 1 − e⁻ˣ).<br><br>
        Quanto mais próxima a curva vermelha estiver da verde, melhor o ajuste.
        Observe que grau {degree} {"captura bem" if degree >= 3 else "ainda não captura" if degree < 2 else "começa a capturar"} a curva do dado real.
    </div>
    """, unsafe_allow_html=True)

    divider()

    secao("⚠️ Underfitting vs Overfitting — o dilema central do ML")

    st.markdown("""
    Escolher o grau correto do polinômio é crucial. Graus muito baixos ou muito altos
    levam a dois problemas opostos — ambos causam modelos ruins para dados novos:
    """)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    graus    = [1, 4, 12]
    cores    = ["orange", "green", "red"]
    rotulos  = ["Underfitting\n(modelo simples demais)", "Ideal\n(bom equilíbrio)", "Overfitting\n(modelo complexo demais)"]

    for ax, d, cor, rot in zip(axes, graus, cores, rotulos):
        m = make_pipeline(PolynomialFeatures(d), LinearRegression())
        m.fit(X_p, y_p)
        y_g  = m.predict(X_grid)
        r2d  = m.score(X_p, y_p)
        ax.scatter(X_p, y_p, color="steelblue", alpha=0.4, s=20, zorder=3)
        ax.plot(X_grid, y_g, color=cor, lw=2.5, zorder=4)
        ax.plot(X_grid, 1 - np.exp(-X_grid), color="black", lw=1.2, linestyle="--", alpha=0.5, label="Real")
        ax.set_title(f"Grau {d} — {rot}\nR² treino = {r2d:.4f}", fontweight="bold", fontsize=10)
        ax.set_xlabel("X")
        ax.grid(True, linestyle="--", alpha=0.3)
    axes[0].set_ylabel("y")
    fig.suptitle("Underfitting vs Ideal vs Overfitting", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_and_show(fig, "07_underfitting_overfitting")

    col_uf1, col_uf2, col_uf3 = st.columns(3)
    with col_uf1:
        caixa_atencao("""
        <b>🟠 Underfitting (grau 1)</b><br><br>
        O modelo é <b>simples demais</b> — a reta não consegue capturar
        a curvatura dos dados.<br><br>
        Erros altos tanto no treino quanto em dados novos.<br><br>
        <i>Analogia: tentar descrever uma estrada sinuosa com uma linha reta.</i>
        """)
    with col_uf2:
        caixa_conceito("""
        <b>🟢 Ideal (grau 4)</b><br><br>
        O modelo captura o padrão real sem exagerar nos detalhes.<br><br>
        Erro razoável no treino e boa generalização para dados novos.<br><br>
        <i>Analogia: mapa com detalhes suficientes para navegar sem ser
        confuso demais.</i>
        """)
    with col_uf3:
        caixa_atencao("""
        <b>🔴 Overfitting (grau 12)</b><br><br>
        O modelo é <b>complexo demais</b> — "decora" os dados de treino
        incluindo o ruído.<br><br>
        R² alto no treino, mas erros grandes em dados novos.<br><br>
        <i>Analogia: memorizar as respostas da prova sem entender o conteúdo.</i>
        """)

    divider()

    with st.expander("📋 Código completo explicado linha a linha"):
        code_block("""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# ─────────────────────────────────────────────────────────────────────────────
# Por que importar PolynomialFeatures?
# Não existe uma classe "PolynomialRegression" no sklearn.
# Em vez disso, usamos PolynomialFeatures para TRANSFORMAR X em [X, X², X³...]
# e depois aplicamos LinearRegression normalmente.
# Isso é um exemplo do conceito de TRANSFORMAÇÃO DE FEATURES — fundamental em ML.
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)
samples = 100

# Dados seguindo y = 1 - e^(-X) — curva de saturação
# Comum em fenômenos físicos: bateria carregando, aprendizado ao longo do tempo, etc.
X = np.random.rand(samples, 1) * 10
e = np.random.randn(samples, 1) * 0.05   # ruído pequeno (σ=0.05)
y = 1 - np.exp(-X) + e

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE: encadeia transformação + modelo em um objeto único
#
# make_pipeline(A, B, C) é equivalente a:
#   X_transformed = A.fit_transform(X)
#   X_final       = B.fit_transform(X_transformed)
#   model         = C.fit(X_final, y)
#
# A grande vantagem: quando chamamos pipeline.predict(X_novo),
# ele aplica TODAS as transformações automaticamente antes de prever.
# ─────────────────────────────────────────────────────────────────────────────
degree = 2

modelo_polinomial = make_pipeline(
    PolynomialFeatures(degree=degree, include_bias=True),
    # include_bias=True → adiciona coluna de 1s (para o intercepto β₀)
    LinearRegression()
)

# .fit() treina o pipeline inteiro:
# 1. PolynomialFeatures transforma X em [1, X, X²]
# 2. LinearRegression encontra β₀, β₁, β₂ que minimizam o erro
modelo_polinomial.fit(X, y)

# ─────────────────────────────────────────────────────────────────────────────
# PREVISÃO COM GRID FINO → curva suave no gráfico
# np.linspace(min, max, 300): 300 pontos igualmente espaçados
# reshape(-1, 1): converte de (300,) para (300, 1) — sklearn exige 2D
# ─────────────────────────────────────────────────────────────────────────────
X_pred = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
y_poly = modelo_polinomial.predict(X_pred)

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color="blue", label="Dados reais", alpha=0.6)
plt.plot(X_pred, y_poly, color="red", lw=2, label=f"Polinômio grau {degree}")
plt.xlabel("X"); plt.ylabel("y")
plt.legend()
plt.show()
""")


# ============================================================================
# PÁGINA: CLASSIFICAÇÃO BINÁRIA
# ============================================================================

elif pagina == "🎯  Classificação Binária":

    st.markdown('<p class="section-title">🎯 Classificação Binária</p>', unsafe_allow_html=True)

    caixa_conceito("""
    <b>Classificação vs Regressão</b><br><br>
    • <b>Regressão</b>: prever um <b>número contínuo</b> (ex: preço, temperatura)<br>
    • <b>Classificação</b>: prever uma <b>categoria</b> (ex: spam/não-spam, tumor benigno/maligno)<br><br>
    Na <b>Classificação Binária</b>, temos exatamente <b>2 classes</b> (0 e 1).<br><br>
    O algoritmo <b>Regressão Logística</b> — apesar do nome confuso — é um classificador.
    Ele calcula a <b>probabilidade</b> de um ponto pertencer à classe 1, usando a função sigmoide:
    <br><br>
    <code>P(y=1 | X) = 1 / (1 + e^(−(β₀ + β₁X₁ + β₂X₂)))</code><br><br>
    Se essa probabilidade for ≥ 0.5, o ponto é classificado como classe 1; caso contrário, classe 0.
    """)

    divider()

    secao("⚙️ Ajuste os parâmetros:")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_class = st.slider("Amostras por classe", 30, 200, 100, step=10)
    with col2:
        sep     = st.slider("Distância entre classes", 1.0, 8.0, 4.0, step=0.5,
                            help="Quanto mais distantes, mais fácil separar")
    with col3:
        spread  = st.slider("Dispersão (σ)", 0.3, 3.0, 1.0, step=0.1,
                            help="Quanto os pontos se espalhiam em torno do centro")
    with col4:
        seed_c  = int(st.number_input("Semente", value=42, step=1))

    np.random.seed(seed_c)
    X0 = np.random.randn(n_class, 2) * spread + np.array([0.0, 0.0])
    X1 = np.random.randn(n_class, 2) * spread + np.array([sep, sep])
    y0 = np.zeros(n_class)
    y1 = np.ones(n_class)

    X_c = np.vstack((X0, X1))
    y_c = np.hstack((y0, y1))

    model_c = LogisticRegression()
    model_c.fit(X_c, y_c)
    acc_c   = accuracy_score(y_c, model_c.predict(X_c))

    divider()

    secao("📊 Como os dados foram construídos?")
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        caixa_conceito(f"""
        <b>Classe 0 (azul)</b><br><br>
        {n_class} pontos gerados com distribuição normal,
        centrados em <code>[0, 0]</code>.<br><br>
        <code>np.random.randn({n_class}, 2) * {spread} + [0, 0]</code><br><br>
        <code>y0 = np.zeros({n_class})</code> → todos recebem rótulo 0.
        """)
    with col_d2:
        caixa_conceito(f"""
        <b>Classe 1 (vermelho)</b><br><br>
        {n_class} pontos gerados com distribuição normal,
        centrados em <code>[{sep}, {sep}]</code>.<br><br>
        <code>np.random.randn({n_class}, 2) * {spread} + [{sep}, {sep}]</code><br><br>
        <code>y1 = np.ones({n_class})</code> → todos recebem rótulo 1.
        """)
    with col_d3:
        caixa_conceito(f"""
        <b>Unindo os datasets</b><br><br>
        <code>np.vstack((X0, X1))</code> → empilha as linhas<br>
        Resultado: X com {2*n_class} linhas e 2 colunas.<br><br>
        <code>np.hstack((y0, y1))</code> → concatena horizontalmente<br>
        Resultado: y com {2*n_class} rótulos (0s e 1s).
        """)

    divider()

    # --- Gráficos ---
    secao("📉 Visualizações")
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("**Dataset: os dois grupos no plano**")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(X0[:, 0], X0[:, 1], c="royalblue", alpha=0.7, edgecolors="k", s=55, label="Classe 0")
        ax.scatter(X1[:, 0], X1[:, 1], c="crimson",   alpha=0.7, edgecolors="k", s=55, label="Classe 1")
        ax.set_title("Dataset de Classificação — 2 Grupos", fontweight="bold")
        ax.set_xlabel("Feature 1 (primeira característica)")
        ax.set_ylabel("Feature 2 (segunda característica)")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        save_and_show(fig, "08_classificacao_dataset")
        st.markdown("""
        <div class="explain-box">
            🔵 <b>Pontos azuis</b>: Classe 0 — centrados em [0, 0].<br>
            🔴 <b>Pontos vermelhos</b>: Classe 1 — centrados em [sep, sep].<br>
            Perceba que as classes formam <b>agrupamentos (clusters)</b>.
            Quanto maior a separação, mais fácil para o modelo distingui-las.
            A sobreposição entre os grupos torna a tarefa mais difícil.
        </div>
        """, unsafe_allow_html=True)

    with col_g2:
        st.markdown("**Fronteira de Decisão: onde o modelo 'traça a linha'**")
        fig = plot_decision_boundary(model_c, X_c, y_c, "Fronteira de Decisão — Regressão Logística")
        save_and_show(fig, "09_fronteira_decisao")
        st.markdown("""
        <div class="explain-box">
            🟥/🟦 <b>Regiões coloridas</b>: onde o modelo classifica como Classe 1 (vermelho)
            ou Classe 0 (azul).<br>
            <b>Fronteira</b>: a linha onde P(y=1|X) = 0.5 — o "empate" entre as duas classes.<br>
            A <b>Regressão Logística sempre gera uma fronteira reta</b> (linear).
            Para fronteiras curvas, precisaríamos de modelos mais complexos.
        </div>
        """, unsafe_allow_html=True)

    divider()

    secao("🔮 Preveja a classe de um novo ponto")
    st.markdown("""
    Mova os sliders para posicionar um ponto no espaço. O modelo dirá qual
    classe ele pertence e com qual **probabilidade**.
    """)

    col_np1, col_np2 = st.columns(2)
    with col_np1:
        px = st.slider("Feature 1 (X)", float(X_c[:, 0].min()), float(X_c[:, 0].max()),
                       float(X_c[:, 0].mean()), step=0.1)
    with col_np2:
        py = st.slider("Feature 2 (Y)", float(X_c[:, 1].min()), float(X_c[:, 1].max()),
                       float(X_c[:, 1].mean()), step=0.1)

    new_point  = [px, py]
    pred_class = int(model_c.predict([new_point])[0])
    prob_class = model_c.predict_proba([new_point])[0]

    col_res1, col_res2, col_res3 = st.columns(3)
    cor_nome = "🔵 Classe 0" if pred_class == 0 else "🔴 Classe 1"
    col_res1.metric("Classe Prevista", cor_nome)
    col_res2.metric("P(Classe 0)", f"{prob_class[0]:.3f} ({prob_class[0]*100:.1f}%)")
    col_res3.metric("P(Classe 1)", f"{prob_class[1]:.3f} ({prob_class[1]*100:.1f}%)")

    caixa_conceito(f"""
    <b>Como interpretar:</b><br><br>
    O ponto ({px:.2f}, {py:.2f}) tem:<br>
    • <b>{prob_class[0]*100:.1f}%</b> de probabilidade de ser Classe 0<br>
    • <b>{prob_class[1]*100:.1f}%</b> de probabilidade de ser Classe 1<br><br>
    Como P(Classe {pred_class}) = {prob_class[pred_class]*100:.1f}% ≥ 50%, o modelo decide:
    <b>Classe {pred_class}</b>.
    """)

    fig = plot_decision_boundary(model_c, X_c, y_c, "Fronteira de Decisão + Novo Ponto")
    ax  = fig.axes[0]
    ax.scatter(px, py, c="yellow", edgecolors="k", s=300, marker="*", zorder=10,
               label=f"Novo ponto → Classe {pred_class}")
    ax.legend()
    save_and_show(fig, "10_classificacao_novo_ponto")
    st.markdown(f"""
    <div class="explain-box">
        ⭐ <b>Estrela amarela</b>: o novo ponto que você escolheu.<br>
        Ele está na região {"azul (Classe 0)" if pred_class == 0 else "vermelha (Classe 1)"} —
        por isso o modelo o classifica como <b>Classe {pred_class}</b>.<br>
        Mova o ponto para a outra região e veja a classificação mudar.
    </div>
    """, unsafe_allow_html=True)

    divider()

    with st.expander("📋 Código completo explicado linha a linha"):
        code_block("""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# ─────────────────────────────────────────────────────────────────────────────
# Por que Regressão LOGÍSTICA para classificação?
# Apesar do nome, é um classificador, não um regressor.
# Ele modela P(y=1|X) — a probabilidade de ser classe 1.
# A função SIGMOIDE garante que a saída fique sempre entre 0 e 1.
# Se P ≥ 0.5 → classe 1; se P < 0.5 → classe 0.
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)

# Classe 0: 100 pontos centrados em [2, 2]
# randn(100, 2): valores com distribuição normal (média=0, σ=1)
# + [2, 2]: desloca o centro para [2, 2]
X0 = np.random.randn(100, 2) + np.array([2, 2])
y0 = np.zeros(100)    # rótulo 0 para todos os pontos desta classe

# Classe 1: 100 pontos centrados em [6, 6]
X1 = np.random.randn(100, 2) + np.array([6, 6])
y1 = np.ones(100)     # rótulo 1 para todos os pontos desta classe

# Unindo os datasets:
# np.vstack: empilha VERTICALMENTE (adiciona linhas)
# np.hstack: concatena HORIZONTALMENTE (adiciona valores ao vetor)
X = np.vstack((X0, X1))   # shape: (200, 2)
y = np.hstack((y0, y1))   # shape: (200,)

# ─────────────────────────────────────────────────────────────────────────────
# TREINAMENTO
# LogisticRegression usa o método LBFGS por padrão (otimização numérica).
# .fit(X, y): encontra os coeficientes β que maximizam a verossimilhança
# (likelihood) — ou seja, que melhor separam as duas classes.
# ─────────────────────────────────────────────────────────────────────────────
model = LogisticRegression()
model.fit(X, y)

# ─────────────────────────────────────────────────────────────────────────────
# PREVISÃO
# .predict([new_point]):   retorna a CLASSE (0 ou 1)
# .predict_proba([...]):   retorna [P(classe 0), P(classe 1)]
# ─────────────────────────────────────────────────────────────────────────────
new_point = [4, 5]
classe  = model.predict([new_point])[0]
probas  = model.predict_proba([new_point])[0]

print(f"Classe prevista: {int(classe)}")
print(f"P(Classe 0) = {probas[0]:.3f}")
print(f"P(Classe 1) = {probas[1]:.3f}")
""")


# ============================================================================
# PÁGINA: DIVISÃO TREINO / TESTE
# ============================================================================

elif pagina == "✂️   Divisão Treino / Teste":

    st.markdown('<p class="section-title">✂️ Divisão Treino e Teste</p>', unsafe_allow_html=True)

    caixa_atencao("""
    <b>O erro mais comum em Machine Learning:</b> avaliar o modelo nos mesmos dados
    em que foi treinado.<br><br>
    Imagine estudar para uma prova com as respostas em mãos — você tiraria 10.
    Mas na prova real, sem as respostas, seu desempenho seria muito menor.<br><br>
    Em ML acontece a mesma coisa: o modelo pode <b>memorizar</b> os dados de treino
    e parecer excelente — mas falhar completamente com dados novos.
    """)

    caixa_conceito("""
    <b>A solução: separar os dados ANTES de treinar</b><br><br>
    • <b>Conjunto de Treino</b>: o modelo aprende com esses dados (vê as "respostas")<br>
    • <b>Conjunto de Teste</b>: simula dados novos — o modelo NUNCA os viu antes<br><br>
    Só avaliamos o desempenho no <b>conjunto de teste</b>. Assim sabemos se o modelo
    realmente aprendeu o padrão ou apenas memorizou.
    """)

    divider()

    secao("⚙️ Ajuste os parâmetros:")
    col1, col2, col3 = st.columns(3)
    with col1:
        n_tot    = st.slider("Total de amostras", 50, 500, 100, step=10)
    with col2:
        test_sz  = st.slider("Proporção de teste (%)", 10, 50, 30, step=5,
                             help="Percentual dos dados reservado para avaliação")
    with col3:
        stratify = st.checkbox("Usar stratify=y", value=True,
                               help="Mantém a proporção de classes nos dois conjuntos")

    np.random.seed(42)
    X_tt = np.random.randn(n_tot, 2) + np.array([2, 2])
    y_tt = np.zeros(n_tot)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tt, y_tt,
        test_size=test_sz / 100,
        random_state=42,
        stratify=y_tt if stratify else None
    )

    n_train = len(X_train)
    n_test  = len(X_test)

    divider()

    secao("📊 Resultado da Divisão")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Total",  f"{n_tot} amostras")
    col_m2.metric("Treino", f"{n_train} ({100-test_sz}%)")
    col_m3.metric("Teste",  f"{n_test} ({test_sz}%)")
    col_m4.metric("Proporção", f"70/30" if test_sz == 30 else f"{100-test_sz}/{test_sz}")

    divider()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(X_train[:, 0], X_train[:, 1], color="royalblue", alpha=0.7,
               edgecolors="k", s=60, label=f"Treino ({n_train} pontos) — o modelo APRENDE aqui")
    ax.scatter(X_test[:, 0],  X_test[:, 1],  color="crimson", alpha=0.7,
               edgecolors="k", s=70, marker="^", label=f"Teste ({n_test} pontos) — o modelo é AVALIADO aqui")
    ax.set_title("Divisão Treino / Teste — Os pontos são os mesmos dados, separados aleatoriamente",
                 fontweight="bold")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    save_and_show(fig, "11_treino_teste")
    st.markdown(f"""
    <div class="explain-box">
        🔵 <b>Círculos azuis</b>: {n_train} amostras de treino — o modelo as vê durante o aprendizado.<br>
        🔺 <b>Triângulos vermelhos</b>: {n_test} amostras de teste — o modelo NUNCA as vê durante o treino.<br><br>
        Os pontos são distribuídos <b>aleatoriamente</b> entre os dois conjuntos.
        O parâmetro <code>random_state=42</code> garante que essa distribuição seja sempre a mesma.
    </div>
    """, unsafe_allow_html=True)

    divider()

    secao("🔍 Entendendo cada parâmetro da função")

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        caixa_conceito("""
        <b><code>test_size=0.3</code></b><br><br>
        Define que <b>30% dos dados</b> vão para o conjunto de teste.<br>
        Os outros 70% vão para treino.<br><br>
        <b>Valores comuns:</b> 0.2 (20%) ou 0.3 (30%)<br><br>
        Quanto mais dados para treino, melhor o modelo aprende.
        Quanto mais para teste, mais confiável a avaliação.
        Um bom equilíbrio é <b>70/30 ou 80/20</b>.
        """)
    with col_p2:
        caixa_conceito("""
        <b><code>random_state=42</code></b><br><br>
        Define a semente <b>local</b> da divisão aleatória.<br><br>
        Diferente do <code>np.random.seed()</code> (que afeta todo o programa),
        o <code>random_state</code> só afeta <i>esta função</i>.<br><br>
        Sem ele, cada execução produziria uma divisão diferente — tornando
        impossível reproduzir os resultados.
        <b>Sempre use!</b>
        """)
    with col_p3:
        caixa_conceito(f"""
        <b><code>stratify=y</code></b><br><br>
        Garante que a <b>proporção de classes</b> seja mantida
        em treino e teste.<br><br>
        Sem stratify: por azar, o teste poderia ter 90% de uma classe
        e 10% de outra — tornando a avaliação não representativa.<br><br>
        Com stratify: se 60% dos dados são classe 0,
        treino e teste também terão 60% de classe 0.<br><br>
        <b>{"✅ Ativado" if stratify else "❌ Desativado"} agora</b>
        """)

    divider()

    secao("📈 Como o tamanho do conjunto de treino afeta o modelo?")

    # Demonstração: accuracy x tamanho do treino (dados IRIS)
    iris   = datasets.load_iris()
    X_ir   = iris.data
    y_ir   = iris.target
    sizes  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    accs   = []
    for ts in sizes:
        Xtr, Xts, ytr, yts = train_test_split(X_ir, y_ir, test_size=ts, random_state=42, stratify=y_ir)
        m = LogisticRegression(max_iter=500)
        m.fit(Xtr, ytr)
        accs.append(accuracy_score(yts, m.predict(Xts)))

    fig, ax = plt.subplots(figsize=(9, 4))
    train_sizes = [1 - s for s in sizes]
    ax.plot([s * 100 for s in train_sizes], accs, marker="o", color="steelblue", lw=2.5, markersize=8)
    ax.fill_between([s * 100 for s in train_sizes], [a - 0.02 for a in accs],
                    [a + 0.02 for a in accs], alpha=0.15, color="steelblue")
    ax.axvline(70, color="green", linestyle="--", lw=1.5, label="70% treino (recomendado)")
    ax.set_xlabel("% dos dados usados para treino")
    ax.set_ylabel("Accuracy no conjunto de teste")
    ax.set_title("Tamanho do Treino vs Desempenho (Dataset IRIS)", fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(0.85, 1.01)
    fig.tight_layout()
    save_and_show(fig, "12_treino_vs_accuracy")
    st.markdown("""
    <div class="explain-box">
        📈 <b>Leitura do gráfico:</b><br>
        Eixo X: percentual dos dados usados para treino (10% a 90%).<br>
        Eixo Y: accuracy obtida no conjunto de teste.<br><br>
        Observe que com muito pouco dado de treino (10-20%), o modelo aprende mal.
        A partir de ~70% de treino, o desempenho se estabiliza.
        Isso explica por que a divisão 70/30 é o padrão mais usado.
    </div>
    """, unsafe_allow_html=True)

    divider()

    with st.expander("📋 Código completo explicado linha a linha"):
        code_block("""
import numpy as np
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────────────
# Por que importar train_test_split?
# Esta função faz a divisão aleatória de forma eficiente e controlada.
# Ela embaralha os dados antes de dividir — garantindo que a distribuição
# das amostras seja similar em treino e teste.
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)
X = np.random.randn(100, 2) + np.array([2, 2])
y = np.zeros(100)

# ─────────────────────────────────────────────────────────────────────────────
# train_test_split(X, y, ...) retorna QUATRO arrays, sempre nesta ordem:
# X_train: features de treino
# X_test:  features de teste   (o modelo NUNCA verá esses durante o treino)
# y_train: rótulos de treino
# y_test:  rótulos de teste    (comparamos com y_pred para avaliar o modelo)
#
# A convenção de nomes X_train, X_test, y_train, y_test é UNIVERSAL em ML.
# Sempre use esses nomes para que outros possam entender seu código.
# ─────────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,      # 30% para teste  →  70 amostras treino, 30 teste
    random_state=42,    # semente para reprodutibilidade desta divisão
    stratify=y          # mantém proporção de classes (crucial para desbalanceados)
)

print(f"Treino: X={X_train.shape}, y={y_train.shape}")
# → Treino: X=(70, 2), y=(70,)

print(f"Teste:  X={X_test.shape},  y={y_test.shape}")
# → Teste:  X=(30, 2), y=(30,)

# ─────────────────────────────────────────────────────────────────────────────
# REGRA FUNDAMENTAL: depois desta linha, X_test e y_test só são usados
# UMA VEZ — na avaliação final. Nunca use dados de teste para ajustar
# parâmetros do modelo (isso se chama "data leakage" e invalida a avaliação).
# ─────────────────────────────────────────────────────────────────────────────
""")


# ============================================================================
# PÁGINA: DATASET IRIS — PIPELINE COMPLETO
# ============================================================================

elif pagina == "🌸  Dataset IRIS — Pipeline Completo":

    st.markdown('<p class="section-title">🌸 Dataset IRIS — Pipeline Completo de ML</p>', unsafe_allow_html=True)

    caixa_conceito("""
    <b>O que é um Pipeline Completo de Machine Learning?</b><br><br>
    Em projetos reais, não fazemos apenas uma etapa — seguimos um fluxo completo:<br><br>
    <b>1. Carregar os dados</b> → <b>2. Explorar e visualizar</b> → <b>3. Dividir treino/teste</b>
    → <b>4. Treinar o modelo</b> → <b>5. Prever</b> → <b>6. Avaliar</b><br><br>
    Nesta seção, seguiremos esse fluxo completo usando o famoso <b>Dataset IRIS</b>,
    introduzido pelo estatístico Ronald Fisher em 1936 — um clássico de ML.
    """)

    divider()

    secao("1️⃣ Carregar e Explorar os Dados")

    iris = datasets.load_iris()
    X_ir = iris.data
    y_ir = iris.target

    st.markdown("""
    O dataset IRIS contém **150 flores** de 3 espécies. Para cada flor,
    foram medidas **4 características** (features):
    """)

    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        caixa_conceito("📏 <b>Sepal length</b><br>Comprimento da sépala (cm)")
    with col_f2:
        caixa_conceito("📏 <b>Sepal width</b><br>Largura da sépala (cm)")
    with col_f3:
        caixa_conceito("🌸 <b>Petal length</b><br>Comprimento da pétala (cm)")
    with col_f4:
        caixa_conceito("🌸 <b>Petal width</b><br>Largura da pétala (cm)")

    col_s1, col_s2, col_s3 = st.columns(3)
    for col, nome, desc in zip([col_s1, col_s2, col_s3],
        ["Iris Setosa (0)", "Iris Versicolor (1)", "Iris Virginica (2)"],
        ["Pétalas pequenas — 50 amostras", "Tamanho médio — 50 amostras", "Pétalas grandes — 50 amostras"]
    ):
        with col:
            st.info(f"**{nome}**\n\n{desc}")

    caixa_dica("""
    <b>Por que o IRIS é tão usado para ensino?</b><br>
    É pequeno (150 amostras), bem organizado, com 3 classes balanceadas (50 cada),
    e representa um problema de classificação real com dificuldade moderada.
    Ideal para aprender sem lidar com problemas de dados grandes ou sujos.
    """)

    # Tabela dos dados
    df_iris = pd.DataFrame(X_ir, columns=iris.feature_names)
    df_iris["espécie"] = [iris.target_names[i] for i in y_ir]
    st.dataframe(df_iris.head(10), use_container_width=True)
    st.caption("Primeiras 10 linhas do dataset — cada linha é uma flor, cada coluna uma característica medida.")

    divider()

    secao("2️⃣ Visualizar os Dados")
    st.markdown("""
    Antes de treinar qualquer modelo, sempre **olhe para os dados**. Gráficos
    revelam padrões, separação entre classes e possíveis problemas.
    """)

    # Parâmetros
    col1, col2 = st.columns(2)
    with col1:
        test_sz_ir = st.slider("Proporção de teste (%)", 10, 40, 30, step=5)
    with col2:
        max_it     = st.slider("max_iter", 100, 1000, 200, step=100,
                               help="Número máximo de iterações do otimizador")

    col_vis1, col_vis2 = st.columns(2)

    with col_vis1:
        X_sep = X_ir[:, :2]
        fig, ax = plt.subplots(figsize=(6, 4))
        sc = ax.scatter(X_sep[:, 0], X_sep[:, 1], c=y_ir, cmap="Set1", edgecolor="k", s=60)
        handles, _ = sc.legend_elements()
        ax.legend(handles, iris.target_names, title="Espécie")
        ax.set_xlabel("Sepal length (cm) →  comprimento da sépala")
        ax.set_ylabel("Sepal width (cm)  →  largura da sépala")
        ax.set_title("IRIS — Features da Sépala", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        save_and_show(fig, "13_iris_sepala")
        st.markdown("""
        <div class="explain-box">
            📊 <b>Leitura:</b> cada ponto é uma flor. As cores indicam a espécie.<br>
            Perceba que a Setosa (vermelho) está bem <b>separada</b> das outras —
            mas Versicolor e Virginica <b>se misturam</b> usando só essas 2 features.<br>
            Isso indica que a sépala sozinha não é suficiente para separar todas as classes.
        </div>
        """, unsafe_allow_html=True)

    with col_vis2:
        X_pet = X_ir[:, 2:4]
        fig, ax = plt.subplots(figsize=(6, 4))
        sc = ax.scatter(X_pet[:, 0], X_pet[:, 1], c=y_ir, cmap="Set1", edgecolor="k", s=60)
        handles, _ = sc.legend_elements()
        ax.legend(handles, iris.target_names, title="Espécie")
        ax.set_xlabel("Petal length (cm) →  comprimento da pétala")
        ax.set_ylabel("Petal width (cm)  →  largura da pétala")
        ax.set_title("IRIS — Features da Pétala", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        save_and_show(fig, "14_iris_petala")
        st.markdown("""
        <div class="explain-box">
            📊 <b>Leitura:</b> usando as features da pétala, a separação é <b>muito melhor</b>!<br>
            Setosa fica completamente isolada. Versicolor e Virginica ainda se tocam um pouco,
            mas são muito mais distinguíveis do que com as features da sépala.<br>
            Isso nos diz que <b>comprimento e largura da pétala são features mais informativas</b>.
        </div>
        """, unsafe_allow_html=True)

    divider()

    secao("3️⃣ Dividir, Treinar e Prever")

    X_train_ir, X_test_ir, y_train_ir, y_test_ir = train_test_split(
        X_ir, y_ir, test_size=test_sz_ir / 100, random_state=42, stratify=y_ir
    )

    model_ir  = LogisticRegression(max_iter=max_it)
    model_ir.fit(X_train_ir, y_train_ir)
    y_pred_ir = model_ir.predict(X_test_ir)

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Amostras de Treino", len(X_train_ir))
    col_m2.metric("Amostras de Teste",  len(X_test_ir))
    col_m3.metric("Accuracy no Teste",  f"{accuracy_score(y_test_ir, y_pred_ir):.4f}")

    caixa_conceito(f"""
    <b>O que acabou de acontecer?</b><br><br>
    1. <code>train_test_split</code>: separou {len(X_train_ir)} flores para treino e {len(X_test_ir)} para teste.<br>
    2. <code>model.fit(X_train, y_train)</code>: o modelo aprendeu os padrões das {len(X_train_ir)} flores de treino.<br>
    3. <code>model.predict(X_test)</code>: o modelo tentou classificar as {len(X_test_ir)} flores que NUNCA viu.<br>
    4. Accuracy = <b>{accuracy_score(y_test_ir, y_pred_ir):.4f}</b> → o modelo acertou
    <b>{int(accuracy_score(y_test_ir, y_pred_ir)*len(X_test_ir))}</b> das <b>{len(X_test_ir)}</b> flores de teste.
    """)

    divider()

    secao("4️⃣ Avaliar o Modelo — Métricas Detalhadas")
    st.markdown("""
    A **accuracy** sozinha não conta toda a história. Usamos métricas mais detalhadas
    para entender onde o modelo acerta e onde erra, para cada classe.
    """)

    col_av1, col_av2 = st.columns(2)

    with col_av1:
        st.markdown("**Classification Report — uma linha por classe**")
        report_dict = classification_report(
            y_test_ir, y_pred_ir,
            target_names=iris.target_names,
            output_dict=True
        )
        df_report = pd.DataFrame(report_dict).T.round(4)
        st.dataframe(df_report, use_container_width=True)

        caixa_conceito("""
        <b>O que significa cada coluna?</b><br><br>
        <b>Precision (Precisão)</b>: dos que o modelo disse "é Setosa",
        quantos REALMENTE eram Setosa?<br>
        <i>Analogia: dos e-mails marcados como spam, quantos eram spam de verdade?</i><br><br>
        <b>Recall (Sensibilidade)</b>: das Setosas reais, quantas o modelo conseguiu encontrar?<br>
        <i>Analogia: de todos os e-mails spam reais, quantos foram detectados?</i><br><br>
        <b>F1-Score</b>: média harmônica de precisão e recall — o equilíbrio entre os dois.<br>
        Use quando precisão e recall são igualmente importantes.<br><br>
        <b>Support</b>: quantas amostras reais existem de cada classe no conjunto de teste.
        """)

    with col_av2:
        st.markdown("**Matriz de Confusão — onde o modelo erra?**")
        cm_ir = confusion_matrix(y_test_ir, y_pred_ir)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm_ir, annot=True, fmt="d", cmap="Blues",
                    xticklabels=iris.target_names,
                    yticklabels=iris.target_names, ax=ax,
                    annot_kws={"size": 14, "weight": "bold"})
        ax.set_xlabel("Previsto pelo modelo →", fontsize=11)
        ax.set_ylabel("← Real (gabarito)", fontsize=11)
        ax.set_title("Matriz de Confusão", fontweight="bold")
        fig.tight_layout()
        save_and_show(fig, "15_iris_confusion_matrix")

        caixa_conceito(f"""
        <b>Como ler a Matriz de Confusão?</b><br><br>
        • <b>Diagonal principal</b> (canto superior esquerdo ao inferior direito):
        os acertos — o modelo previu a classe correta.<br>
        • <b>Fora da diagonal</b>: os erros — o modelo confundiu uma classe com outra.<br><br>
        <b>Exemplo:</b> se há um número na linha "versicolor" × coluna "virginica",
        significa que o modelo classificou flores versicolor como virginica esse número de vezes.<br><br>
        Quanto mais concentrado na diagonal, melhor o modelo.
        """)

    divider()

    secao("5️⃣ Fronteira de Decisão — Visualização em 2D")
    st.markdown("""
    Para visualizar a fronteira de decisão, precisamos reduzir para 2 features.
    Escolha quais features usar:
    """)

    feat_options = {
        "Sépala: comprimento × largura (features 0 e 1)":  (0, 1, "Sepal length (cm)", "Sepal width (cm)"),
        "Pétala: comprimento × largura (features 2 e 3)":  (2, 3, "Petal length (cm)", "Petal width (cm)"),
        "Comprimento sépala × comprimento pétala (0 e 2)": (0, 2, "Sepal length (cm)", "Petal length (cm)"),
        "Largura sépala × largura pétala (1 e 3)":         (1, 3, "Sepal width (cm)",  "Petal width (cm)"),
    }
    feat_sel = st.selectbox("Features para o gráfico de fronteira:", list(feat_options.keys()))
    fi, fj, xlabel, ylabel = feat_options[feat_sel]

    X_vis_ir = X_ir[:, [fi, fj]]
    Xtv, Xtsv, ytv, ytsv = train_test_split(
        X_vis_ir, y_ir, test_size=test_sz_ir / 100, random_state=42, stratify=y_ir
    )
    m_vis = LogisticRegression(max_iter=max_it)
    m_vis.fit(Xtv, ytv)
    acc_vis = accuracy_score(ytsv, m_vis.predict(Xtsv))

    fig = plot_decision_boundary(m_vis, X_vis_ir, y_ir, "")
    ax  = fig.axes[0]
    ax.set_xlabel(f"{xlabel} →")
    ax.set_ylabel(f"← {ylabel}")
    ax.set_title(
        f"Fronteira de Decisão: {feat_sel.split('(')[0].strip()}\n"
        f"Accuracy no teste = {acc_vis:.4f}  |  Usando apenas 2 das 4 features",
        fontweight="bold"
    )
    save_and_show(fig, "16_iris_fronteira")
    st.markdown(f"""
    <div class="explain-box">
        🔴/🔵/⚪ <b>Regiões coloridas</b>: zonas onde o modelo classifica cada espécie.<br>
        <b>Fronteiras</b>: linhas onde o modelo muda de decisão (P = 0.5).<br><br>
        Com apenas 2 features, a accuracy cai para <b>{acc_vis:.4f}</b> (contra
        {accuracy_score(y_test_ir, y_pred_ir):.4f} com todas as 4 features).<br>
        Isso demonstra que <b>mais features informativas = melhor separação das classes</b>.<br><br>
        Troque as features no menu acima e veja como a fronteira muda!
    </div>
    """, unsafe_allow_html=True)

    divider()

    with st.expander("📋 Código completo explicado linha a linha"):
        code_block("""
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 1: Carregar o dataset
# datasets.load_iris() carrega o IRIS já pronto, sem precisar de arquivo.
# O sklearn inclui vários datasets clássicos para aprendizado.
#
# iris.data:          matriz (150, 4) — 150 flores, 4 medidas cada
# iris.target:        vetor (150,)   — 0=Setosa, 1=Versicolor, 2=Virginica
# iris.feature_names: nomes das 4 colunas (strings)
# iris.target_names:  nomes das 3 classes
# ─────────────────────────────────────────────────────────────────────────────
iris = datasets.load_iris()
X = iris.data
y = iris.target

print(f"Shape de X: {X.shape}")  # (150, 4)
print(f"Shape de y: {y.shape}")  # (150,)
print(f"Features: {iris.feature_names}")
print(f"Classes:  {iris.target_names}")

# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 2: Dividir em treino e teste
# stratify=y: essencial para manter 50 flores de cada classe em proporção.
# Sem isso, o conjunto de teste poderia ter desequilíbrio por azar.
# ─────────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
# Resultado: 105 flores para treino, 45 para teste (35/15 de cada classe)

# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 3: Treinar o modelo
# max_iter=200: o otimizador LBFGS precisa de mais iterações que o padrão (100)
# para convergir no IRIS. Sem isso, aparece ConvergenceWarning.
# ─────────────────────────────────────────────────────────────────────────────
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)   # TREINA: X_train e y_train — nunca X_test!

# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 4: Previsão — só agora usamos X_test!
# ─────────────────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
print(f"Primeiras 10 previsões: {y_pred[:10]}")
print(f"Primeiros 10 reais:     {y_test[:10]}")

# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 5: Avaliação completa
# ─────────────────────────────────────────────────────────────────────────────

# Accuracy: proporção de acertos (entre 0 e 1)
# Ex: 0.9556 → acertou 43 das 45 flores de teste
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")

# Classification Report: métricas por classe
# precision, recall, f1-score para cada uma das 3 espécies
report = classification_report(y_test, y_pred, target_names=iris.target_names)
print(report)

# Confusion Matrix: tabela de acertos e erros por classe
# cm[i][j] = número de flores da classe i classificadas como classe j
# Diagonal principal = acertos; fora = erros
cm = confusion_matrix(y_test, y_pred)

# Heatmap com seaborn: visualização colorida da matriz de confusão
# annot=True: mostra os números dentro das células
# fmt="d": formato inteiro (não decimal)
# cmap="Blues": escala de cores azul (mais escuro = maior valor)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel("Previsto pelo modelo")
plt.ylabel("Real (gabarito)")
plt.title("Matriz de Confusão — IRIS")
plt.show()
""")


# ============================================================================
# MATERIAL DO ALUNO — DOWNLOAD DO NOTEBOOK
# ============================================================================

st.markdown("---")
st.markdown("### 📓 Material do Aluno")

_notebook_path = os.path.join(os.path.dirname(__file__), "Aula_01_Introducao_ao_ML_(aluno).ipynb")
if os.path.exists(_notebook_path):
    with open(_notebook_path, "rb") as _f:
        st.download_button(
            label="⬇️ Baixar notebook do aluno — Aula 01 (versão para praticar)",
            data=_f,
            file_name="Aula_01_Introducao_ao_ML_(aluno).ipynb",
            mime="application/json",
            use_container_width=True,
        )
    st.info(
        "💡 Este notebook contém os mesmos conteúdos da aula com células de exercício em branco. "
        "Abra no Jupyter e complete os exercícios marcados com ✍️."
    )


# ============================================================================
# FOOTER — ASSINATURA
# ============================================================================

st.markdown("---")
st.markdown(
    "<div style='text-align:center; padding:1rem 0; color:#888; font-size:0.85rem;'>"
    "Cláudio Ferreira Neves &nbsp;·&nbsp; Especialista em Ciência de Dados e IA"
    "</div>",
    unsafe_allow_html=True,
)
