"""
=============================================================================
Machine Learning — Aula 04 / Regressão Logística e KNN
Aplicação Streamlit interativa — material didático para iniciantes

Autor       : Cláudio Ferreira Neves
Cargo atual : Analista de BI — Save Co. / Especialista de Ensino II — SENAI/SC
Certificação: DATA ANALYST CERTIFIED PROFESSIONAL © (DACP)
=============================================================================
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# URLs de navegação entre aulas — ajuste as portas se necessário
# ---------------------------------------------------------------------------
PAGE_PORTAL  = "pages/Portal.py"
PAGE_AULA_03 = "pages/Aula_03.py"
PAGE_AULA_05 = "pages/Aula_05.py"

import matplotlib
matplotlib.use("Agg")   # Backend não-interativo: impede abertura de janelas externas
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

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
def carregar_diabetes() -> pd.DataFrame:
    """Carrega o dataset Diabetes do GitHub."""
    url = (
        "https://raw.githubusercontent.com/matheusvanzan/"
        "Machine-Learning-Examples/refs/heads/master/datasets/diabetes.csv"
    )
    return pd.read_csv(url)


@st.cache_data
def carregar_penguins() -> pd.DataFrame:
    """Carrega e prepara o dataset Penguins do seaborn."""
    df = sns.load_dataset("penguins")
    df.columns = [
        "espécie", "ilha", "comprimento_bico_mm", "profundidade_bico_mm",
        "comprimento_nadadeira_mm", "massa_corporal_g", "sexo"
    ]
    df["sexo"] = df["sexo"].map({"Male": "macho", "Female": "fêmea"})
    df = df.dropna().reset_index(drop=True)
    return df


# ============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="Regressão Logística e KNN",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS — visual premium (mesmo padrão das aulas anteriores)
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
        <h2 style="margin:0; font-size:1.3rem;">🎯 Aula 04</h2>
        <p style="margin:0.4rem 0 0; font-size:0.8rem; opacity:0.8;">Regressão Logística e KNN</p>
    </div>
    """, unsafe_allow_html=True)

    pagina = st.radio(
        "Navegação",
        options=[
            "🏠  Início",
            "📈  Sigmoid e Regressão Logística",
            "🩺  Dataset Diabetes",
            "🐧  Dataset Penguins (multi-classe)",
            "📏  KNN — K-Nearest Neighbors",
            "🔍  Otimização do K",
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
tab_nav(4)  # replace CURRENT_NUMBER with the correct int

st.markdown("<p style='text-align:center; font-size:0.95rem; color:#667eea; font-weight:600; margin-bottom:0;'>Autor: Especialista Cláudio Ferreira Neves</p>", unsafe_allow_html=True)

_nav_l, _nav_m, _nav_r = st.columns([1.2, 4, 1.2])
with _nav_l:
    if st.button("← Aula 03", use_container_width=True, key="nav_prev"):
        st.switch_page(PAGE_AULA_03)
with _nav_m:
    if st.button("🏠 Portal", use_container_width=True, key="nav_portal"):
        st.switch_page(PAGE_PORTAL)
with _nav_r:
    if st.button("Aula 05 →", use_container_width=True, key="nav_next"):
        st.switch_page(PAGE_AULA_05)


# ============================================================================
# PÁGINA: INÍCIO
# ============================================================================

if pagina == "🏠  Início":

    st.markdown("""
    <div style='text-align:center; padding: 2rem 0 1rem;'>
        <h1 style='font-size:2.8rem; font-weight:800; color:#1a1a2e; margin-bottom:0.3rem;'>
            🎯 Regressão Logística e KNN
        </h1>
        <p style='font-size:1.15rem; color:#555; max-width:720px; margin:0 auto;'>
            Classificar é decidir — do diagnóstico de diabetes à espécie de pinguim.
            Aprenda dois dos algoritmos de classificação mais usados em Machine Learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    divider()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><h2>2</h2><p>Algoritmos</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h2>2</h2><p>Datasets</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h2>🎯</h2><p>Classificação</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><h2>📊</h2><p>Cross-Val</p></div>', unsafe_allow_html=True)

    divider()

    st.markdown("## 📚 O que você vai aprender")
    col_a, col_b = st.columns(2)

    with col_a:
        caixa_conceito("""
        <b>📈 Regressão Logística</b><br>
        Apesar do nome, é um algoritmo de <b>classificação</b>. Usa a função sigmoid
        para converter uma combinação linear de features em uma probabilidade entre 0 e 1.
        Ideal para problemas binários e multi-classe.
        """)
        caixa_conceito("""
        <b>🩺 Dataset Diabetes</b><br>
        768 pacientes com features clínicas (glicose, pressão, IMC...).
        Target binário: tem diabetes ou não. Demonstração completa de Pipeline
        com imputação de zeros, escalonamento e validação cruzada.
        """)
        caixa_conceito("""
        <b>📊 Cross-Validation</b><br>
        Validação cruzada com cv=10 dobras para estimar a performance real
        do modelo. Métricas reportadas como média ± desvio padrão.
        """)

    with col_b:
        caixa_conceito("""
        <b>📏 K-Nearest Neighbors (KNN)</b><br>
        Algoritmo intuitivo: classifica um novo ponto pela classe majoritária
        entre seus k vizinhos mais próximos. Sem etapa de treinamento explícita
        — toda a computação ocorre na predição.
        """)
        caixa_conceito("""
        <b>🐧 Dataset Penguins</b><br>
        Palmer Penguins com 3 espécies: Adelie, Chinstrap e Gentoo.
        Problema multi-classe com features numéricas e categóricas.
        Demonstração da estratégia One-vs-Rest da Regressão Logística.
        """)
        caixa_conceito("""
        <b>🔍 Otimização do K</b><br>
        Analisar como a escolha de k afeta o tradeoff viés-variância.
        Curvas de Accuracy e F1 vs k para encontrar o ponto ótimo
        em ambos os datasets.
        """)

    divider()
    st.markdown("## 🧰 Bibliotecas utilizadas")

    col_lib1, col_lib2, col_lib3 = st.columns(3)
    with col_lib1:
        caixa_dica("<b>scikit-learn</b> — <code>LogisticRegression</code>, <code>KNeighborsClassifier</code>, "
                   "<code>Pipeline</code>, <code>ColumnTransformer</code>, <code>SimpleImputer</code>")
    with col_lib2:
        caixa_dica("<b>pandas / numpy</b> — manipulação de dados, cálculos numéricos e "
                   "preparação dos datasets para modelagem.")
    with col_lib3:
        caixa_dica("<b>matplotlib / seaborn</b> — visualizações interativas: histogramas, "
                   "heatmaps, scatter plots e curvas de métricas.")

    divider()
    st.markdown("""
    <div class="footer">
        <p style="margin:0; font-size:1.1rem; font-weight:700;">Machine Learning — Aula 04</p>
        <p style="margin:0.3rem 0 0; font-size:0.9rem; opacity:0.8;">
            Regressão Logística e KNN &nbsp;·&nbsp;
            Cláudio Ferreira Neves &nbsp;·&nbsp; SENAI/SC
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# PÁGINA: SIGMOID E REGRESSÃO LOGÍSTICA
# ============================================================================

elif pagina == "📈  Sigmoid e Regressão Logística":

    st.markdown('<p class="section-title">📈 Função Sigmoid e Regressão Logística</p>',
                unsafe_allow_html=True)
    divider()

    caixa_conceito("""
    <b>O que é Regressão Logística?</b><br>
    Apesar do nome, é um algoritmo de <b>classificação</b>, não regressão.
    Ela modela a <b>probabilidade</b> de uma observação pertencer a uma classe,
    usando a função sigmoid para mapear qualquer número real para o intervalo (0, 1).
    """)

    divider()
    secao("A Função Sigmoid σ(z)")

    col_eq, col_slider = st.columns([1, 1])

    with col_eq:
        st.markdown(r"""
        **Fórmula:**

        $$\sigma(z) = \frac{1}{1 + e^{-z}}$$

        Onde $z$ é a combinação linear das features:

        $$z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n$$

        **Propriedades:**
        - σ(z) → 0 quando z → -∞
        - σ(z) → 1 quando z → +∞
        - σ(0) = 0.5 (ponto de decisão)
        - Derivada: σ'(z) = σ(z) · (1 - σ(z))
        """)

        caixa_conceito("""
        <b>Regra de decisão:</b><br>
        • Se σ(z) ≥ 0.5 → <b>classe 1</b> (z ≥ 0)<br>
        • Se σ(z) < 0.5 → <b>classe 0</b> (z < 0)<br>
        O limiar 0.5 é chamado de <b>fronteira de decisão</b>.
        """)

    with col_slider:
        z_escolhido = st.slider(
            "Escolha o valor de z:", min_value=-10.0, max_value=10.0,
            value=0.0, step=0.1
        )
        sigma_z = 1 / (1 + np.exp(-z_escolhido))
        classe_prevista = 1 if sigma_z >= 0.5 else 0

        st.metric("σ(z)", f"{sigma_z:.4f}")
        st.metric("Classe prevista", str(classe_prevista))

        if sigma_z >= 0.5:
            st.success(f"σ({z_escolhido:.1f}) = {sigma_z:.4f} → **CLASSE 1** (positivo)")
        else:
            st.error(f"σ({z_escolhido:.1f}) = {sigma_z:.4f} → **CLASSE 0** (negativo)")

    # Gráfico sigmoid interativo
    z_vals = np.linspace(-10, 10, 500)
    sig_vals = 1 / (1 + np.exp(-z_vals))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(z_vals, sig_vals, color="#667eea", linewidth=2.5,
            label="σ(z) = 1/(1+e⁻ᶻ)")
    ax.axhline(0.5, color="tomato", linestyle="--", linewidth=1.5,
               label="Limiar de decisão = 0.5")
    ax.axvline(z_escolhido, color="gold", linestyle="-", linewidth=2,
               label=f"z = {z_escolhido:.1f}")
    ax.scatter([z_escolhido], [sigma_z], color="gold", s=200, zorder=5,
               label=f"σ({z_escolhido:.1f}) = {sigma_z:.4f}")
    ax.fill_between(z_vals, sig_vals, 0.5, where=(sig_vals >= 0.5),
                    alpha=0.12, color="seagreen", label="Região Classe 1")
    ax.fill_between(z_vals, sig_vals, 0.5, where=(sig_vals < 0.5),
                    alpha=0.12, color="tomato", label="Região Classe 0")
    ax.set_xlabel("z (combinação linear das features)", fontsize=11)
    ax.set_ylabel("σ(z) — probabilidade", fontsize=11)
    ax.set_title("Função Sigmoid — Regressão Logística", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    save_and_show(fig, "sigmoid_interativa")

    divider()
    secao("Conceitos fundamentais")

    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        caixa_conceito("""
        <b>Log-Loss (Binary Cross-Entropy)</b><br>
        Função de custo da Regressão Logística:<br>
        <code>J = -1/n Σ [y·log(ŷ) + (1-y)·log(1-ŷ)]</code><br>
        Penaliza erros de forma assimétrica: prever 0.01 para uma amostra
        que é classe 1 recebe punição enorme.
        """)
    with col_c2:
        caixa_conceito("""
        <b>Log-Odds (Logit)</b><br>
        A Regressão Logística modela os log-odds:<br>
        <code>log(p/(1-p)) = β₀ + β₁x₁ + ... + βₙxₙ</code><br>
        Cada coeficiente βⱼ muda as odds por um fator de e^(βⱼ) quando
        xⱼ aumenta 1 unidade.
        """)
    with col_c3:
        caixa_conceito("""
        <b>Regularização L2 (Ridge)</b><br>
        O parâmetro <code>penalty='l2'</code> adiciona uma penalidade
        proporcional ao quadrado dos coeficientes:<br>
        <code>J_reg = J + λ·Σβⱼ²</code><br>
        Evita overfitting ao restringir a magnitude dos pesos.
        """)

    divider()
    caixa_atencao("""
    <b>Regressão Logística ≠ Regressão Linear</b><br>
    • Linear: saída é um número contínuo (ex.: preço, temperatura)<br>
    • Logística: saída é uma probabilidade [0, 1] → decisão de classe<br>
    A função sigmoid <b>comprime</b> qualquer valor z para o intervalo (0, 1),
    tornando a saída interpretável como probabilidade.
    """)

    divider()
    with st.expander("Ver código Python — Sigmoid e Regressão Logística"):
        code_block("""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Função sigmoid manual
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Valores de demonstração
for z in [-5, -2, -1, 0, 1, 2, 5]:
    print(f"σ({z:+2d}) = {sigmoid(z):.4f}")

# Treinando um modelo
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(penalty="l2", max_iter=1000, random_state=42)),
])
pipe.fit(X_train, y_train)
probas = pipe.predict_proba(X_test)   # probabilidades por classe
y_pred = pipe.predict(X_test)         # classes preditas (threshold=0.5)
""")


# ============================================================================
# PÁGINA: DATASET DIABETES
# ============================================================================

elif pagina == "🩺  Dataset Diabetes":

    st.markdown('<p class="section-title">🩺 Dataset Diabetes — Classificação Binária</p>',
                unsafe_allow_html=True)
    divider()

    caixa_conceito("""
    <b>Pima Indians Diabetes Database</b><br>
    768 pacientes do sexo feminino (≥21 anos) de origem Pima.
    Target: <code>Outcome</code> (0 = sem diabetes, 1 = com diabetes).<br>
    Features clínicas: Pregnancies, Glucose, BloodPressure, SkinThickness,
    Insulin, BMI, DiabetesPedigreeFunction, Age.
    """)

    with st.spinner("Carregando dataset Diabetes..."):
        try:
            df_diab = carregar_diabetes()
            st.success(f"Dataset carregado: {df_diab.shape[0]} linhas × {df_diab.shape[1]} colunas")
        except Exception as e:
            st.error(f"Erro ao carregar dataset: {e}")
            st.stop()

    # EDA
    secao("Análise Exploratória")

    col_eda1, col_eda2 = st.columns(2)
    with col_eda1:
        st.markdown("**Primeiras linhas:**")
        st.dataframe(df_diab.head(8), use_container_width=True)
    with col_eda2:
        st.markdown("**Distribuição do Target:**")
        vc = df_diab["Outcome"].value_counts().rename({0: "Sem Diabetes", 1: "Com Diabetes"})
        st.dataframe(vc.reset_index().rename(columns={"index": "Classe", "Outcome": "Contagem"}),
                     use_container_width=True)
        pct_pos = df_diab["Outcome"].mean() * 100
        st.info(f"Positivos (diabetes): **{pct_pos:.1f}%**")

    st.markdown("**Estatísticas descritivas:**")
    st.dataframe(df_diab.describe().round(2), use_container_width=True)

    divider()
    secao("Zeros como valores ausentes")
    caixa_atencao("""
    Colunas clínicas como Glucose, BloodPressure, SkinThickness, Insulin e BMI
    não podem ter valor 0 biologicamente. Esses zeros representam <b>dados faltantes</b>
    e precisam ser imputados antes do treinamento.
    Usamos <code>SimpleImputer(missing_values=0, strategy='median')</code>.
    """)

    cols_impute = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    zero_counts = {c: (df_diab[c] == 0).sum() for c in cols_impute}
    df_zeros = pd.DataFrame({
        "Coluna": list(zero_counts.keys()),
        "Zeros": list(zero_counts.values()),
        "% do total": [v / len(df_diab) * 100 for v in zero_counts.values()],
    })
    st.dataframe(df_zeros.style.format({"% do total": "{:.1f}%"}),
                 use_container_width=True)

    divider()
    secao("Visualizações")

    col_v1, col_v2 = st.columns(2)

    with col_v1:
        st.markdown("**Histograma de Glicose por Diagnóstico**")
        fig, ax = plt.subplots(figsize=(6, 4))
        for outcome, color, label in [
            (0, "steelblue", "Sem diabetes"),
            (1, "tomato", "Com diabetes")
        ]:
            ax.hist(df_diab[df_diab["Outcome"] == outcome]["Glucose"],
                    bins=30, alpha=0.65, color=color, label=label)
        ax.set_xlabel("Glucose")
        ax.set_ylabel("Frequência")
        ax.set_title("Glicose por Diagnóstico")
        ax.legend()
        ax.grid(alpha=0.3)
        save_and_show(fig, "diabetes_glucose_hist")

    with col_v2:
        st.markdown("**Heatmap de Correlação**")
        fig, ax = plt.subplots(figsize=(7, 5))
        corr = df_diab.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, ax=ax, linewidths=0.5, annot_kws={"size": 8})
        ax.set_title("Correlação — Diabetes")
        save_and_show(fig, "diabetes_correlacao")

    divider()
    secao("Pipeline: Imputação + Escalonamento + Regressão Logística")

    caixa_conceito("""
    <b>Arquitetura do Pipeline:</b><br>
    1. <code>ColumnTransformer</code> com dois ramos:<br>
       &nbsp;&nbsp;• Colunas com zeros: <code>SimpleImputer(median) → StandardScaler</code><br>
       &nbsp;&nbsp;• Demais colunas numéricas: <code>StandardScaler</code><br>
    2. <code>LogisticRegression(penalty='l2', max_iter=1000)</code>
    """)

    # Slider para test_size interativo
    test_size_d = st.slider(
        "Proporção do conjunto de teste:", min_value=0.1, max_value=0.4,
        value=0.2, step=0.05, key="ts_diab",
        help="Altere para ver como o tamanho do conjunto de teste afeta as métricas."
    )

    X_diab = df_diab.drop("Outcome", axis=1)
    y_diab = df_diab["Outcome"]
    feat_names_diab = X_diab.columns.tolist()

    X_tr_d, X_te_d, y_tr_d, y_te_d = train_test_split(
        X_diab, y_diab, test_size=test_size_d, random_state=42, stratify=y_diab
    )

    preprocessor_diab = ColumnTransformer(transformers=[
        ("imp_scl", Pipeline([
            ("imp", SimpleImputer(missing_values=0, strategy="median")),
            ("scl", StandardScaler()),
        ]), cols_impute),
        ("scl_only", StandardScaler(),
         [c for c in feat_names_diab if c not in cols_impute]),
    ])

    pipe_diab = Pipeline([
        ("pre", preprocessor_diab),
        ("clf", LogisticRegression(penalty="l2", max_iter=1000, random_state=42)),
    ])
    pipe_diab.fit(X_tr_d, y_tr_d)
    y_pred_d = pipe_diab.predict(X_te_d)

    acc_d  = accuracy_score(y_te_d, y_pred_d)
    f1_d   = f1_score(y_te_d, y_pred_d, average="macro")

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Accuracy", f"{acc_d:.4f}")
    col_m2.metric("F1-macro", f"{f1_d:.4f}")
    col_m3.metric("Amostras teste", len(y_te_d))

    divider()
    secao("Relatório de Classificação")

    report_d = classification_report(y_te_d, y_pred_d,
                                     target_names=["Sem Diabetes", "Com Diabetes"],
                                     output_dict=True)
    df_report_d = pd.DataFrame(report_d).T.round(4)
    st.dataframe(df_report_d, use_container_width=True)

    secao("Matriz de Confusão")
    cm_d = confusion_matrix(y_te_d, y_pred_d)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_d, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Sem Diabetes", "Com Diabetes"],
                yticklabels=["Sem Diabetes", "Com Diabetes"],
                linewidths=1)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão — Diabetes")
    save_and_show(fig, "diabetes_confusion_matrix_lr")

    divider()
    secao("Validação Cruzada (cv=10)")
    caixa_conceito("""
    A validação cruzada divide os dados em 10 dobras iguais, treina em 9 e testa
    em 1, repetindo 10 vezes. O resultado final é a média e o desvio padrão das
    10 avaliações — estimativa mais confiável do que um único split treino/teste.
    """)

    with st.spinner("Executando cross-validation (cv=10)..."):
        cv_d = cross_validate(
            pipe_diab, X_diab, y_diab, cv=10,
            scoring=["accuracy", "f1_macro"]
        )
    col_cv1, col_cv2 = st.columns(2)
    col_cv1.metric("CV Accuracy",
                   f"{cv_d['test_accuracy'].mean():.4f} ± {cv_d['test_accuracy'].std():.4f}")
    col_cv2.metric("CV F1-macro",
                   f"{cv_d['test_f1_macro'].mean():.4f} ± {cv_d['test_f1_macro'].std():.4f}")

    # Gráfico de dobras
    fig, ax = plt.subplots(figsize=(9, 4))
    dobras = range(1, 11)
    ax.bar([d - 0.2 for d in dobras], cv_d["test_accuracy"],
           width=0.35, label="Accuracy", color="#667eea", alpha=0.85)
    ax.bar([d + 0.2 for d in dobras], cv_d["test_f1_macro"],
           width=0.35, label="F1-macro", color="#764ba2", alpha=0.85)
    ax.axhline(cv_d["test_accuracy"].mean(), color="#667eea", linestyle="--",
               linewidth=1.5, alpha=0.7)
    ax.axhline(cv_d["test_f1_macro"].mean(), color="#764ba2", linestyle="--",
               linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Dobra (fold)")
    ax.set_ylabel("Métrica")
    ax.set_title("Cross-Validation cv=10 — Diabetes (Regressão Logística)")
    ax.set_xticks(list(dobras))
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    save_and_show(fig, "diabetes_cv10")

    divider()
    with st.expander("Ver código Python — Dataset Diabetes"):
        code_block("""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import classification_report, confusion_matrix

URL = "https://raw.githubusercontent.com/matheusvanzan/Machine-Learning-Examples/refs/heads/master/datasets/diabetes.csv"
df = pd.read_csv(URL)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

cols_impute = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
cols_scale  = ["Pregnancies", "DiabetesPedigreeFunction", "Age"]

preprocessor = ColumnTransformer(transformers=[
    ("imp_scl", Pipeline([
        ("imp", SimpleImputer(missing_values=0, strategy="median")),
        ("scl", StandardScaler()),
    ]), cols_impute),
    ("scl_only", StandardScaler(), cols_scale),
])

pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", LogisticRegression(penalty="l2", max_iter=1000, random_state=42)),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred))

# Cross-validation
cv = cross_validate(pipe, X, y, cv=10, scoring=["accuracy", "f1_macro"])
print(f"Accuracy: {cv['test_accuracy'].mean():.4f} ± {cv['test_accuracy'].std():.4f}")
""")


# ============================================================================
# PÁGINA: DATASET PENGUINS
# ============================================================================

elif pagina == "🐧  Dataset Penguins (multi-classe)":

    st.markdown('<p class="section-title">🐧 Dataset Penguins — Classificação Multi-classe</p>',
                unsafe_allow_html=True)
    divider()

    caixa_atencao("""
    <b>Classificação Multi-classe:</b> a Regressão Logística nativa é binária,
    mas o scikit-learn a estende para múltiplas classes usando a estratégia
    <b>One-vs-Rest (OvR)</b>: treina um classificador binário para cada classe
    ("Adelie vs resto", "Chinstrap vs resto", "Gentoo vs resto")
    e seleciona a classe com maior probabilidade.
    """)

    with st.spinner("Carregando dataset Penguins..."):
        try:
            df_pen = carregar_penguins()
            st.success(f"Dataset carregado: {df_pen.shape[0]} pinguins × {df_pen.shape[1]} colunas")
        except Exception as e:
            st.error(f"Erro ao carregar dataset: {e}")
            st.stop()

    secao("Análise Exploratória")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("**Distribuição por espécie:**")
        vc_pen = df_pen["espécie"].value_counts().reset_index()
        vc_pen.columns = ["Espécie", "Contagem"]
        st.dataframe(vc_pen, use_container_width=True)
        st.markdown("**Distribuição por ilha:**")
        vi_pen = df_pen["ilha"].value_counts().reset_index()
        vi_pen.columns = ["Ilha", "Contagem"]
        st.dataframe(vi_pen, use_container_width=True)
    with col_p2:
        st.markdown("**Primeiras linhas:**")
        st.dataframe(df_pen.head(8), use_container_width=True)

    divider()
    secao("Visualizações")

    col_vp1, col_vp2 = st.columns(2)

    with col_vp1:
        st.markdown("**Scatter: Comprimento vs Profundidade do Bico**")
        fig, ax = plt.subplots(figsize=(6, 5))
        paleta = {"Adelie": "#4C72B0", "Chinstrap": "#DD8452", "Gentoo": "#55A868"}
        for esp, color in paleta.items():
            sub = df_pen[df_pen["espécie"] == esp]
            ax.scatter(sub["comprimento_bico_mm"], sub["profundidade_bico_mm"],
                       label=esp, color=color, alpha=0.65, s=30)
        ax.set_xlabel("Comprimento do Bico (mm)")
        ax.set_ylabel("Profundidade do Bico (mm)")
        ax.set_title("Bico — Penguins por Espécie")
        ax.legend()
        ax.grid(alpha=0.3)
        save_and_show(fig, "penguins_scatter_bico")

    with col_vp2:
        st.markdown("**Heatmap de Correlação Numérica**")
        num_pen_cols = ["comprimento_bico_mm", "profundidade_bico_mm",
                        "comprimento_nadadeira_mm", "massa_corporal_g"]
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(df_pen[num_pen_cols].corr(), annot=True, fmt=".2f",
                    cmap="coolwarm", center=0, ax=ax, linewidths=0.5,
                    annot_kws={"size": 10})
        ax.set_title("Correlação Numérica — Penguins")
        save_and_show(fig, "penguins_correlacao")

    st.markdown("**Boxplot: Massa Corporal por Espécie**")
    fig, ax = plt.subplots(figsize=(8, 4))
    df_pen.boxplot(column="massa_corporal_g", by="espécie", ax=ax,
                   patch_artist=True)
    ax.set_xlabel("Espécie")
    ax.set_ylabel("Massa Corporal (g)")
    ax.set_title("Massa Corporal por Espécie")
    plt.suptitle("")
    ax.grid(alpha=0.3, axis="y")
    save_and_show(fig, "penguins_boxplot_massa")

    divider()
    secao("Pipeline: Pré-processamento + Regressão Logística Multi-classe")

    caixa_conceito("""
    <b>Pré-processamento:</b><br>
    • Features numéricas: <code>StandardScaler</code><br>
    • Features categóricas (ilha, sexo): <code>OneHotEncoder(drop='first')</code><br>
    <b>Modelo:</b> <code>LogisticRegression(max_iter=1000)</code> — usa OvR automaticamente.
    """)

    X_pen = df_pen.drop("espécie", axis=1)
    y_pen = df_pen["espécie"]
    num_pen = ["comprimento_bico_mm", "profundidade_bico_mm",
               "comprimento_nadadeira_mm", "massa_corporal_g"]
    cat_pen = ["ilha", "sexo"]

    X_tr_p, X_te_p, y_tr_p, y_te_p = train_test_split(
        X_pen, y_pen, test_size=0.2, stratify=y_pen, random_state=25
    )

    pre_pen = ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_pen),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_pen),
    ])
    pipe_pen = Pipeline([
        ("pre", pre_pen),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipe_pen.fit(X_tr_p, y_tr_p)
    y_pred_p = pipe_pen.predict(X_te_p)

    acc_p = accuracy_score(y_te_p, y_pred_p)
    f1_p  = f1_score(y_te_p, y_pred_p, average="macro")

    col_mp1, col_mp2, col_mp3 = st.columns(3)
    col_mp1.metric("Accuracy", f"{acc_p:.4f}")
    col_mp2.metric("F1-macro", f"{f1_p:.4f}")
    col_mp3.metric("Amostras teste", len(y_te_p))

    divider()
    secao("Relatório de Classificação")
    report_p = classification_report(y_te_p, y_pred_p, output_dict=True)
    df_report_p = pd.DataFrame(report_p).T.round(4)
    st.dataframe(df_report_p, use_container_width=True)

    secao("Matriz de Confusão")
    classes_pen = sorted(y_pen.unique())
    cm_p = confusion_matrix(y_te_p, y_pred_p, labels=classes_pen)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_p, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=classes_pen, yticklabels=classes_pen, linewidths=1)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão — Penguins (Regressão Logística)")
    save_and_show(fig, "penguins_confusion_matrix_lr")

    divider()
    secao("Validação Cruzada (cv=10)")

    with st.spinner("Executando cross-validation (cv=10)..."):
        cv_p = cross_validate(
            pipe_pen, X_pen, y_pen, cv=10,
            scoring=["accuracy", "f1_macro"]
        )
    col_cvp1, col_cvp2 = st.columns(2)
    col_cvp1.metric("CV Accuracy",
                    f"{cv_p['test_accuracy'].mean():.4f} ± {cv_p['test_accuracy'].std():.4f}")
    col_cvp2.metric("CV F1-macro",
                    f"{cv_p['test_f1_macro'].mean():.4f} ± {cv_p['test_f1_macro'].std():.4f}")

    fig, ax = plt.subplots(figsize=(9, 4))
    dobras = range(1, 11)
    ax.bar([d - 0.2 for d in dobras], cv_p["test_accuracy"],
           width=0.35, label="Accuracy", color="#38a169", alpha=0.85)
    ax.bar([d + 0.2 for d in dobras], cv_p["test_f1_macro"],
           width=0.35, label="F1-macro", color="#2c7a7b", alpha=0.85)
    ax.axhline(cv_p["test_accuracy"].mean(), color="#38a169", linestyle="--",
               linewidth=1.5, alpha=0.7)
    ax.axhline(cv_p["test_f1_macro"].mean(), color="#2c7a7b", linestyle="--",
               linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Dobra (fold)")
    ax.set_ylabel("Métrica")
    ax.set_title("Cross-Validation cv=10 — Penguins (Regressão Logística)")
    ax.set_xticks(list(dobras))
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    save_and_show(fig, "penguins_cv10")

    divider()
    with st.expander("Ver código Python — Dataset Penguins"):
        code_block("""
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import classification_report

# Carregamento e renomeação
df = sns.load_dataset("penguins")
df.columns = ["espécie","ilha","comprimento_bico_mm","profundidade_bico_mm",
              "comprimento_nadadeira_mm","massa_corporal_g","sexo"]
df["sexo"] = df["sexo"].map({"Male": "macho", "Female": "fêmea"})
df = df.dropna().reset_index(drop=True)

X = df.drop("espécie", axis=1)
y = df["espécie"]

num_cols = ["comprimento_bico_mm","profundidade_bico_mm",
            "comprimento_nadadeira_mm","massa_corporal_g"]
cat_cols = ["ilha","sexo"]

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols),
])

pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", LogisticRegression(max_iter=1000, random_state=42)),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=25
)
pipe.fit(X_train, y_train)
print(classification_report(y_test, pipe.predict(X_test)))

# Cross-validation
cv = cross_validate(pipe, X, y, cv=10, scoring=["accuracy","f1_macro"])
print(f"Accuracy: {cv['test_accuracy'].mean():.4f} ± {cv['test_accuracy'].std():.4f}")
""")


# ============================================================================
# PÁGINA: KNN — K-NEAREST NEIGHBORS
# ============================================================================

elif pagina == "📏  KNN — K-Nearest Neighbors":

    st.markdown('<p class="section-title">📏 KNN — K-Nearest Neighbors</p>',
                unsafe_allow_html=True)
    divider()

    caixa_conceito("""
    <b>Como funciona o KNN?</b><br>
    1. Para classificar um novo ponto, calcule a distância para todos os pontos de treino.<br>
    2. Selecione os <b>k</b> pontos mais próximos (vizinhos).<br>
    3. A classe prevista é a <b>maioria de votos</b> entre os k vizinhos.<br><br>
    KNN é um algoritmo <b>preguiçoso (lazy learner)</b>: não constrói um modelo explícito.
    Toda a computação acontece no momento da predição.
    """)

    secao("Distância Euclidiana")
    col_dist1, col_dist2 = st.columns(2)
    with col_dist1:
        st.markdown(r"""
        **Distância Euclidiana (padrão do KNN):**

        $$d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$

        Para 2 dimensões:
        $$d = \sqrt{(x_2-x_1)^2 + (y_2-y_1)^2}$$
        """)
    with col_dist2:
        caixa_atencao("""
        <b>Importância do escalonamento:</b><br>
        O KNN é altamente sensível à escala das features. Uma feature com
        valores na casa dos milhares domina a distância em relação a features
        na casa das unidades. Sempre use <code>StandardScaler</code> antes do KNN.
        """)

    divider()
    secao("Demonstração Interativa — Fronteira de Decisão")

    k_demo = st.slider(
        "Escolha o valor de k:", min_value=1, max_value=15,
        value=3, step=2, key="knn_demo_k",
        help="Valores ímpares evitam empate na votação."
    )

    np.random.seed(99)
    n_pts = 40
    X_demo_0 = np.random.randn(n_pts, 2) + np.array([-1.5, 0])
    X_demo_1 = np.random.randn(n_pts, 2) + np.array([1.5, 0])
    X_demo   = np.vstack([X_demo_0, X_demo_1])
    y_demo   = np.array([0] * n_pts + [1] * n_pts)
    novo_ponto = np.array([[0.2, 0.5]])

    knn_demo = KNeighborsClassifier(n_neighbors=k_demo)
    knn_demo.fit(X_demo, y_demo)
    pred_demo = knn_demo.predict(novo_ponto)[0]
    proba_demo = knn_demo.predict_proba(novo_ponto)[0]

    h = 0.05
    x_min, x_max = X_demo[:, 0].min() - 1, X_demo[:, 0].max() + 1
    y_min_m, y_max_m = X_demo[:, 1].min() - 1, X_demo[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min_m, y_max_m, h))
    Z = knn_demo.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Encontrar vizinhos mais próximos
    dists = np.sqrt(((X_demo - novo_ponto) ** 2).sum(axis=1))
    idx_vizinhos = np.argsort(dists)[:k_demo]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.contourf(xx, yy, Z, alpha=0.18, cmap="RdBu")
    ax.scatter(X_demo_0[:, 0], X_demo_0[:, 1], color="#4C72B0", s=40,
               alpha=0.75, label="Classe 0", zorder=3)
    ax.scatter(X_demo_1[:, 0], X_demo_1[:, 1], color="#DD8452", s=40,
               alpha=0.75, label="Classe 1", zorder=3)
    ax.scatter(novo_ponto[0, 0], novo_ponto[0, 1], color="gold",
               s=250, marker="*", zorder=6,
               label=f"Novo ponto → Classe {pred_demo}")
    for idx_v in idx_vizinhos:
        ax.plot([novo_ponto[0, 0], X_demo[idx_v, 0]],
                [novo_ponto[0, 1], X_demo[idx_v, 1]],
                "k--", linewidth=0.9, alpha=0.5)
        ax.scatter(X_demo[idx_v, 0], X_demo[idx_v, 1],
                   edgecolors="black", facecolors="none", s=120, linewidths=1.5, zorder=5)
    ax.set_title(f"KNN — k={k_demo} | Novo ponto → Classe {pred_demo} "
                 f"(P(0)={proba_demo[0]:.2f}, P(1)={proba_demo[1]:.2f})",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    save_and_show(fig, f"knn_demo_k{k_demo}")

    col_kd1, col_kd2 = st.columns(2)
    col_kd1.metric("Classe Predita", str(pred_demo))
    col_kd2.metric(f"P(Classe {pred_demo})", f"{proba_demo[pred_demo]:.4f}")

    divider()
    secao("KNN no Dataset Diabetes (k=3) e Penguins (k=5)")

    with st.spinner("Treinando KNN nos datasets..."):
        try:
            df_diab_knn = carregar_diabetes()
            X_dk = df_diab_knn.drop("Outcome", axis=1)
            y_dk = df_diab_knn["Outcome"]
            cols_imp_knn = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
            cols_scl_knn = ["Pregnancies", "DiabetesPedigreeFunction", "Age"]

            pipe_knn_d = Pipeline([
                ("pre", ColumnTransformer(transformers=[
                    ("imp_scl", Pipeline([
                        ("imp", SimpleImputer(missing_values=0, strategy="median")),
                        ("scl", StandardScaler()),
                    ]), cols_imp_knn),
                    ("scl_only", StandardScaler(), cols_scl_knn),
                ])),
                ("clf", KNeighborsClassifier(n_neighbors=3)),
            ])
            pipe_lr_d = Pipeline([
                ("pre", ColumnTransformer(transformers=[
                    ("imp_scl", Pipeline([
                        ("imp", SimpleImputer(missing_values=0, strategy="median")),
                        ("scl", StandardScaler()),
                    ]), cols_imp_knn),
                    ("scl_only", StandardScaler(), cols_scl_knn),
                ])),
                ("clf", LogisticRegression(penalty="l2", max_iter=1000, random_state=42)),
            ])

            X_tr_dk, X_te_dk, y_tr_dk, y_te_dk = train_test_split(
                X_dk, y_dk, test_size=0.2, random_state=42, stratify=y_dk
            )
            pipe_knn_d.fit(X_tr_dk, y_tr_dk)
            pipe_lr_d.fit(X_tr_dk, y_tr_dk)
            y_pred_knn_d  = pipe_knn_d.predict(X_te_dk)
            y_pred_lr_d   = pipe_lr_d.predict(X_te_dk)
            acc_knn_d = accuracy_score(y_te_dk, y_pred_knn_d)
            f1_knn_d  = f1_score(y_te_dk, y_pred_knn_d, average="macro")
            acc_lr_d  = accuracy_score(y_te_dk, y_pred_lr_d)
            f1_lr_d   = f1_score(y_te_dk, y_pred_lr_d, average="macro")
            diab_ok = True
        except Exception as e:
            st.error(f"Erro Diabetes: {e}")
            diab_ok = False

        try:
            df_pen_knn = carregar_penguins()
            X_pk = df_pen_knn.drop("espécie", axis=1)
            y_pk = df_pen_knn["espécie"]
            num_pk = ["comprimento_bico_mm", "profundidade_bico_mm",
                      "comprimento_nadadeira_mm", "massa_corporal_g"]
            cat_pk = ["ilha", "sexo"]

            pipe_knn_p = Pipeline([
                ("pre", ColumnTransformer(transformers=[
                    ("num", StandardScaler(), num_pk),
                    ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_pk),
                ])),
                ("clf", KNeighborsClassifier(n_neighbors=5)),
            ])
            pipe_lr_p = Pipeline([
                ("pre", ColumnTransformer(transformers=[
                    ("num", StandardScaler(), num_pk),
                    ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_pk),
                ])),
                ("clf", LogisticRegression(max_iter=1000, random_state=42)),
            ])

            X_tr_pk, X_te_pk, y_tr_pk, y_te_pk = train_test_split(
                X_pk, y_pk, test_size=0.2, stratify=y_pk, random_state=25
            )
            pipe_knn_p.fit(X_tr_pk, y_tr_pk)
            pipe_lr_p.fit(X_tr_pk, y_tr_pk)
            y_pred_knn_p = pipe_knn_p.predict(X_te_pk)
            y_pred_lr_p  = pipe_lr_p.predict(X_te_pk)
            acc_knn_p = accuracy_score(y_te_pk, y_pred_knn_p)
            f1_knn_p  = f1_score(y_te_pk, y_pred_knn_p, average="macro")
            acc_lr_p  = accuracy_score(y_te_pk, y_pred_lr_p)
            f1_lr_p   = f1_score(y_te_pk, y_pred_lr_p, average="macro")
            pen_ok = True
        except Exception as e:
            st.error(f"Erro Penguins: {e}")
            pen_ok = False

    divider()
    secao("Comparação: Regressão Logística vs KNN")

    if diab_ok and pen_ok:
        df_comp = pd.DataFrame({
            "Dataset":    ["Diabetes", "Diabetes", "Penguins", "Penguins"],
            "Modelo":     ["LogisticRegression", "KNN (k=3)",
                           "LogisticRegression", "KNN (k=5)"],
            "Accuracy":   [f"{acc_lr_d:.4f}", f"{acc_knn_d:.4f}",
                           f"{acc_lr_p:.4f}", f"{acc_knn_p:.4f}"],
            "F1-macro":   [f"{f1_lr_d:.4f}", f"{f1_knn_d:.4f}",
                           f"{f1_lr_p:.4f}", f"{f1_knn_p:.4f}"],
        })
        st.dataframe(df_comp, use_container_width=True)

        # Gráfico comparativo
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        modelos = ["LR", "KNN"]

        for ax_i, (dataset, acc_lr, acc_knn, f1_lr, f1_knn) in enumerate([
            ("Diabetes", acc_lr_d, acc_knn_d, f1_lr_d, f1_knn_d),
            ("Penguins", acc_lr_p, acc_knn_p, f1_lr_p, f1_knn_p),
        ]):
            x_pos = [0, 1]
            axes[ax_i].bar([p - 0.18 for p in x_pos], [acc_lr, acc_knn],
                           width=0.32, label="Accuracy", color="#667eea", alpha=0.9)
            axes[ax_i].bar([p + 0.18 for p in x_pos], [f1_lr, f1_knn],
                           width=0.32, label="F1-macro", color="#764ba2", alpha=0.9)
            axes[ax_i].set_xticks(x_pos)
            axes[ax_i].set_xticklabels(modelos)
            axes[ax_i].set_ylim(0, 1.05)
            axes[ax_i].set_ylabel("Métrica")
            axes[ax_i].set_title(f"LR vs KNN — {dataset}")
            axes[ax_i].legend(fontsize=9)
            axes[ax_i].grid(alpha=0.3, axis="y")
            # Anotar valores
            for p_i, (acc_v, f1_v) in enumerate([(acc_lr, f1_lr), (acc_knn, f1_knn)]):
                axes[ax_i].text(p_i - 0.18, acc_v + 0.01, f"{acc_v:.3f}",
                                ha="center", fontsize=8)
                axes[ax_i].text(p_i + 0.18, f1_v + 0.01, f"{f1_v:.3f}",
                                ha="center", fontsize=8)
        save_and_show(fig, "comparacao_lr_knn")

    divider()
    col_knn_c1, col_knn_c2 = st.columns(2)
    with col_knn_c1:
        caixa_conceito("""
        <b>Métricas de Distância no KNN</b><br>
        • <b>Euclidiana</b>: raiz da soma dos quadrados (padrão)<br>
        • <b>Manhattan (L1)</b>: soma dos valores absolutos<br>
        • <b>Minkowski</b>: generalização de Euclidiana e Manhattan<br>
        • <b>Cosine</b>: ângulo entre vetores (texto e NLP)
        """)
    with col_knn_c2:
        caixa_atencao("""
        <b>Maldição da Dimensionalidade</b><br>
        Em alta dimensão, todos os pontos tendem a ficar igualmente
        distantes entre si. O conceito de "vizinhança" perde significado.
        Soluções: padronização das features, seleção de features e PCA.
        """)

    divider()
    with st.expander("Ver código Python — KNN"):
        code_block("""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Pipeline KNN para Diabetes
cols_imp = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
cols_scl = ["Pregnancies", "DiabetesPedigreeFunction", "Age"]

pipe_knn = Pipeline([
    ("pre", ColumnTransformer(transformers=[
        ("imp_scl", Pipeline([
            ("imp", SimpleImputer(missing_values=0, strategy="median")),
            ("scl", StandardScaler()),
        ]), cols_imp),
        ("scl_only", StandardScaler(), cols_scl),
    ])),
    ("clf", KNeighborsClassifier(n_neighbors=3)),
])

pipe_knn.fit(X_train, y_train)
y_pred = pipe_knn.predict(X_test)
print(classification_report(y_test, y_pred))
""")


# ============================================================================
# PÁGINA: OTIMIZAÇÃO DO K
# ============================================================================

elif pagina == "🔍  Otimização do K":

    st.markdown('<p class="section-title">🔍 Otimização do K — Tradeoff Viés-Variância</p>',
                unsafe_allow_html=True)
    divider()

    caixa_conceito("""
    <b>Como k afeta o modelo?</b><br>
    • <b>k pequeno (k=1)</b>: o modelo se ajusta a cada ponto de treino individualmente.
      <b>Alta variância</b> — fronteiras de decisão muito irregulares (overfitting).<br>
    • <b>k grande</b>: o modelo usa uma vizinhança ampla, suavizando a fronteira.
      <b>Alto viés</b> — pode não capturar padrões locais (underfitting).<br>
    • <b>k ótimo</b>: equilíbrio entre viés e variância, encontrado via cross-validation.
    """)

    caixa_dica("""
    Use <b>cross-validation</b> para encontrar o melhor k, nunca apenas a acurácia
    no conjunto de treino (que sempre favorece k=1). Use k ímpar para evitar empates.
    """)

    divider()
    secao("Curva Accuracy e F1 vs K — Dataset Diabetes")

    with st.spinner("Calculando métricas para k = 3 a 300 (pode demorar alguns segundos)..."):
        try:
            df_diab_opt = carregar_diabetes()
            X_dopt = df_diab_opt.drop("Outcome", axis=1)
            y_dopt = df_diab_opt["Outcome"]
            cols_imp_opt = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
            cols_scl_opt = ["Pregnancies", "DiabetesPedigreeFunction", "Age"]

            k_range = list(range(3, 301, 10))
            accs_d_opt, f1s_d_opt = [], []

            for k_val in k_range:
                pipe_kopt = Pipeline([
                    ("pre", ColumnTransformer(transformers=[
                        ("imp_scl", Pipeline([
                            ("imp", SimpleImputer(missing_values=0, strategy="median")),
                            ("scl", StandardScaler()),
                        ]), cols_imp_opt),
                        ("scl_only", StandardScaler(), cols_scl_opt),
                    ])),
                    ("clf", KNeighborsClassifier(n_neighbors=k_val)),
                ])
                cv_kopt = cross_validate(
                    pipe_kopt, X_dopt, y_dopt, cv=5,
                    scoring=["accuracy", "f1_macro"]
                )
                accs_d_opt.append(cv_kopt["test_accuracy"].mean())
                f1s_d_opt.append(cv_kopt["test_f1_macro"].mean())

            best_k_acc_d = k_range[int(np.argmax(accs_d_opt))]
            best_k_f1_d  = k_range[int(np.argmax(f1s_d_opt))]
            diab_opt_ok = True
        except Exception as e:
            st.error(f"Erro: {e}")
            diab_opt_ok = False

    if diab_opt_ok:
        col_bk1, col_bk2 = st.columns(2)
        col_bk1.metric("Melhor k por Accuracy", f"k = {best_k_acc_d}",
                       f"{max(accs_d_opt):.4f}")
        col_bk2.metric("Melhor k por F1-macro", f"k = {best_k_f1_d}",
                       f"{max(f1s_d_opt):.4f}")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(k_range, accs_d_opt, "b-o", markersize=5, linewidth=2,
                label="Accuracy (cv=5)")
        ax.plot(k_range, f1s_d_opt, "r-s", markersize=5, linewidth=2,
                label="F1-macro (cv=5)")
        ax.axvline(best_k_acc_d, color="blue", linestyle="--", linewidth=1.5,
                   alpha=0.65, label=f"Melhor k (Acc) = {best_k_acc_d}")
        ax.axvline(best_k_f1_d, color="red", linestyle="--", linewidth=1.5,
                   alpha=0.65, label=f"Melhor k (F1) = {best_k_f1_d}")
        ax.scatter([best_k_acc_d], [max(accs_d_opt)], color="blue", s=150,
                   marker="*", zorder=5)
        ax.scatter([best_k_f1_d], [max(f1s_d_opt)], color="red", s=150,
                   marker="*", zorder=5)
        ax.set_xlabel("k (n_neighbors)")
        ax.set_ylabel("Métrica média (cv=5)")
        ax.set_title("Otimização do K — Dataset Diabetes")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        save_and_show(fig, "otimizacao_k_diabetes")

    divider()
    secao("Curva Accuracy e F1 vs K — Dataset Penguins")

    with st.spinner("Calculando métricas para Penguins..."):
        try:
            df_pen_opt = carregar_penguins()
            X_popt = df_pen_opt.drop("espécie", axis=1)
            y_popt = df_pen_opt["espécie"]
            num_popt = ["comprimento_bico_mm", "profundidade_bico_mm",
                        "comprimento_nadadeira_mm", "massa_corporal_g"]
            cat_popt = ["ilha", "sexo"]

            accs_p_opt, f1s_p_opt = [], []

            for k_val in k_range:
                pipe_kpopt = Pipeline([
                    ("pre", ColumnTransformer(transformers=[
                        ("num", StandardScaler(), num_popt),
                        ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_popt),
                    ])),
                    ("clf", KNeighborsClassifier(n_neighbors=k_val)),
                ])
                cv_kpopt = cross_validate(
                    pipe_kpopt, X_popt, y_popt, cv=5,
                    scoring=["accuracy", "f1_macro"]
                )
                accs_p_opt.append(cv_kpopt["test_accuracy"].mean())
                f1s_p_opt.append(cv_kpopt["test_f1_macro"].mean())

            best_k_acc_p = k_range[int(np.argmax(accs_p_opt))]
            best_k_f1_p  = k_range[int(np.argmax(f1s_p_opt))]
            pen_opt_ok = True
        except Exception as e:
            st.error(f"Erro: {e}")
            pen_opt_ok = False

    if pen_opt_ok:
        col_bkp1, col_bkp2 = st.columns(2)
        col_bkp1.metric("Melhor k por Accuracy", f"k = {best_k_acc_p}",
                        f"{max(accs_p_opt):.4f}")
        col_bkp2.metric("Melhor k por F1-macro", f"k = {best_k_f1_p}",
                        f"{max(f1s_p_opt):.4f}")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(k_range, accs_p_opt, "b-o", markersize=5, linewidth=2,
                label="Accuracy (cv=5)")
        ax.plot(k_range, f1s_p_opt, "r-s", markersize=5, linewidth=2,
                label="F1-macro (cv=5)")
        ax.axvline(best_k_acc_p, color="blue", linestyle="--", linewidth=1.5,
                   alpha=0.65, label=f"Melhor k (Acc) = {best_k_acc_p}")
        ax.axvline(best_k_f1_p, color="red", linestyle="--", linewidth=1.5,
                   alpha=0.65, label=f"Melhor k (F1) = {best_k_f1_p}")
        ax.scatter([best_k_acc_p], [max(accs_p_opt)], color="blue", s=150,
                   marker="*", zorder=5)
        ax.scatter([best_k_f1_p], [max(f1s_p_opt)], color="red", s=150,
                   marker="*", zorder=5)
        ax.set_xlabel("k (n_neighbors)")
        ax.set_ylabel("Métrica média (cv=5)")
        ax.set_title("Otimização do K — Dataset Penguins")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        save_and_show(fig, "otimizacao_k_penguins")

    divider()
    secao("Interpretação do Tradeoff Viés-Variância")

    col_bv1, col_bv2, col_bv3 = st.columns(3)
    with col_bv1:
        caixa_atencao("""
        <b>k pequeno → Overfitting</b><br>
        O modelo memoriza os dados de treino. Alta variância:
        pequenas mudanças nos dados mudam muito as predições.
        Fronteiras de decisão muito irregulares.
        """)
    with col_bv2:
        caixa_conceito("""
        <b>k ideal → Equilíbrio</b><br>
        Encontrado via cross-validation. Generaliza bem para
        dados novos sem perder a capacidade de capturar
        padrões locais nos dados.
        """)
    with col_bv3:
        caixa_atencao("""
        <b>k grande → Underfitting</b><br>
        O modelo usa uma vizinhança muito ampla, ignorando
        estrutura local. Alto viés: decisões baseadas em
        regiões grandes e homogêneas.
        """)

    divider()
    with st.expander("Ver código Python — Otimização do K"):
        code_block("""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

k_range = range(3, 301, 10)
accs, f1s = [], []

for k in k_range:
    pipe = Pipeline([
        ("pre", preprocessor),   # ColumnTransformer já definido
        ("clf", KNeighborsClassifier(n_neighbors=k)),
    ])
    cv = cross_validate(pipe, X, y, cv=5, scoring=["accuracy", "f1_macro"])
    accs.append(cv["test_accuracy"].mean())
    f1s.append(cv["test_f1_macro"].mean())

best_k_acc = list(k_range)[np.argmax(accs)]
best_k_f1  = list(k_range)[np.argmax(f1s)]
print(f"Melhor k por Accuracy: {best_k_acc} ({max(accs):.4f})")
print(f"Melhor k por F1-macro: {best_k_f1}  ({max(f1s):.4f})")

import matplotlib.pyplot as plt
plt.plot(list(k_range), accs, "b-o", label="Accuracy")
plt.plot(list(k_range), f1s,  "r-s", label="F1-macro")
plt.axvline(best_k_acc, color="blue", linestyle="--")
plt.xlabel("k")
plt.ylabel("Métrica (cv=5)")
plt.title("Otimização do K")
plt.legend()
plt.show()
""")

    divider()
    st.markdown("""
    <div class="footer">
        <p style="margin:0; font-size:1.1rem; font-weight:700;">Machine Learning — Aula 04</p>
        <p style="margin:0.3rem 0 0; font-size:0.9rem; opacity:0.8;">
            Regressão Logística e KNN &nbsp;·&nbsp;
            Cláudio Ferreira Neves &nbsp;·&nbsp; SENAI/SC
        </p>
    </div>
    """, unsafe_allow_html=True)
