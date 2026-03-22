"""
=============================================================================
Machine Learning — Aula 05 / Naive Bayes e SVM
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

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score,
)
from sklearn.datasets import load_wine, make_blobs, make_moons

PAGE_PORTAL  = "pages/Portal.py"
PAGE_AULA_04 = "pages/Aula_04.py"
PAGE_AULA_06 = "pages/Aula_06.py"

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ── helpers ─────────────────────────────────────────────────────────────────
def save_and_show(fig, filename):
    fig.savefig(os.path.join(OUTPUTS_DIR, f"{filename}.png"), dpi=150, bbox_inches="tight")
    st.pyplot(fig); plt.close(fig)

def caixa(html, cor="#f0f4ff", borda="#667eea"):
    st.markdown(f'<div style="background:{cor};border-left:5px solid {borda};'
                f'padding:1rem 1.5rem;border-radius:0 10px 10px 0;margin:.8rem 0;">'
                f'{html}</div>', unsafe_allow_html=True)

def divider(): st.markdown("---")
def secao(t): st.markdown(f"### {t}")

def code_block(code, title=""):
    if title: st.markdown(f"**{title}**")
    st.code(code, language="python")

# ── dados ────────────────────────────────────────────────────────────────────
@st.cache_data
def carregar_penguins():
    df = sns.load_dataset("penguins")
    df.columns = ["espécie","ilha","comprimento_bico_mm","profundidade_bico_mm",
                  "comprimento_nadadeira_mm","massa_corporal_g","sexo"]
    df["sexo"] = df["sexo"].map({"Male":"macho","Female":"fêmea"})
    return df.dropna().reset_index(drop=True)

@st.cache_data
def carregar_wine():
    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df, data.target_names

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Naive Bayes e SVM", page_icon="🧠",
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

# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-header"><h2 style="margin:0;font-size:1.3rem;">🧠 Aula 05</h2>'
                '<p style="margin:.4rem 0 0;font-size:.8rem;opacity:.8;">Naive Bayes e SVM</p></div>',
                unsafe_allow_html=True)
    pagina = st.radio("Nav", [
        "🏠  Início",
        "📊  Naive Bayes — Teoria",
        "🐧  Naive Bayes — Pinguins",
        "⚔️  SVM — Teoria e Kernels",
        "🐧  SVM — Pinguins + GridSearch",
        "🍷  Exercício — Dataset Wine",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("<div style='font-size:.75rem;color:#888;text-align:center;'>"
                "📁 Gráficos salvos em <code>outputs/</code></div>", unsafe_allow_html=True)

# ── autor + nav ───────────────────────────────────────────────────────────────
# ── navegação entre aulas ────────────────────────────────────────────────────
from nav import tab_nav
tab_nav(5)  # replace CURRENT_NUMBER with the correct int

st.markdown("<p style='text-align:center;font-size:.95rem;color:#667eea;font-weight:600;margin-bottom:0;'>"
            "Autor: Especialista Cláudio Ferreira Neves</p>", unsafe_allow_html=True)
nl, nm, nr = st.columns([1.2, 4, 1.2])
with nl:
    if st.button("← Aula 04", use_container_width=True, key="nav_prev"):
        st.switch_page(PAGE_AULA_04)
with nm:
    if st.button("🏠 Portal", use_container_width=True, key="nav_portal"):
        st.switch_page(PAGE_PORTAL)
with nr:
    if st.button("Aula 06 →", use_container_width=True, key="nav_next"):
        st.switch_page(PAGE_AULA_06)

# ============================================================================
# INÍCIO
# ============================================================================
if pagina == "🏠  Início":
    st.markdown("<div style='text-align:center;padding:2rem 0 1rem;'>"
                "<h1 style='font-size:2.8rem;font-weight:800;color:#1a1a2e;'>🧠 Naive Bayes e SVM</h1>"
                "<p style='font-size:1.15rem;color:#555;max-width:720px;margin:0 auto;'>"
                "Dois classificadores com abordagens opostas: um baseado em probabilidade,"
                " outro em geometria — ambos poderosos na prática.</p></div>",
                unsafe_allow_html=True)
    divider()
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown('<div class="metric-card"><h2>2</h2><p>Algoritmos</p></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="metric-card"><h2>2</h2><p>Datasets</p></div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="metric-card"><h2>🔍</h2><p>GridSearch</p></div>', unsafe_allow_html=True)
    with c4: st.markdown('<div class="metric-card"><h2>🍷</h2><p>Exercício Wine</p></div>', unsafe_allow_html=True)
    divider()
    st.markdown("## 📚 O que você vai aprender")
    ca, cb = st.columns(2)
    with ca:
        caixa("<b>📊 Naive Bayes (GaussianNB)</b><br>Classificador probabilístico baseado no "
              "<b>Teorema de Bayes</b>. Assume independência entre features (hipótese "
              '"ingênua"). Rápido, simples e surpreendentemente eficaz — especialmente para '
              "dados contínuos com distribuição aproximadamente normal.")
        caixa("<b>🤔 Por que NÃO usar OneHotEncoder com GNB?</b><br>O GaussianNB assume "
              "distribuição <b>normal contínua</b> em cada feature. OneHotEncoder gera "
              "valores binários (0/1) que violam essa suposição. Aprenda a contornar isso "
              "com <code>remainder='drop'</code>.", cor="#fff8e1", borda="#f59e0b")
        caixa("<b>🔍 GridSearchCV</b><br>Busca exaustiva de hiperparâmetros com "
              "validação cruzada estratificada. Testa <b>todas as combinações</b> "
              "do <code>param_grid</code> e retorna o melhor modelo.")
    with cb:
        caixa("<b>⚔️ SVM — Support Vector Machine</b><br>Busca o <b>hiperplano ótimo</b> "
              "que maximiza a <b>margem</b> entre as classes. Os pontos mais próximos "
              "da fronteira são os <b>vetores de suporte</b>. Suporta dados não lineares "
              "via <b>kernels</b> (linear, RBF, poly).")
        caixa("<b>🎛️ Parâmetros do SVM</b><br>"
              "• <b>C</b>: penalidade por erros (C alto = margem estreita = risco de overfitting)<br>"
              "• <b>kernel</b>: linear (reto) ou RBF (curvo)<br>"
              "• <b>γ (gamma)</b>: alcance da curvatura no RBF")
        caixa("<b>🍷 Exercício — Dataset Wine</b><br>3 tipos de vinho, 13 features químicas. "
              "Compare KNN, GNB e SVM com GridSearch para encontrar o melhor modelo.")
    divider()
    st.markdown("""<div class="footer">
        <p style="margin:0;font-size:1.1rem;font-weight:700;">Machine Learning — Aula 05</p>
        <p style="margin:.3rem 0 0;font-size:.9rem;opacity:.8;">
        Naive Bayes e SVM · Cláudio Ferreira Neves · SENAI/SC</p></div>""",
        unsafe_allow_html=True)

# ============================================================================
# NAIVE BAYES — TEORIA
# ============================================================================
elif pagina == "📊  Naive Bayes — Teoria":
    st.markdown('<h2>📊 Naive Bayes — Fundamentos</h2>', unsafe_allow_html=True)
    divider()

    caixa("<b>O que é Naive Bayes?</b><br>Família de classificadores probabilísticos baseados "
          "no <b>Teorema de Bayes</b>. Fazem a hipótese <i>ingênua</i> de que as features são "
          "<b>condicionalmente independentes</b> entre si, dado o rótulo de classe. Raramente "
          "isso é 100% verdadeiro, mas funciona muito bem na prática.")

    divider()
    secao("O Teorema de Bayes")
    ct1, ct2 = st.columns(2)
    with ct1:
        st.markdown(r"""
**Fórmula:**
$$P(C \mid X) = \frac{P(X \mid C) \cdot P(C)}{P(X)}$$

- $P(C \mid X)$ = probabilidade **a posteriori** da classe dado os dados
- $P(X \mid C)$ = **verossimilhança** — chance de observar X na classe C  
- $P(C)$ = **probabilidade a priori** da classe  
- $P(X)$ = **evidência** (constante, serve como normalizador)

**Hipótese de independência:**
$$P(X \mid C) = \prod_{i=1}^{n} P(x_i \mid C)$$

**Regra de decisão:**
$$\hat{C} = \arg\max_C \; P(C) \cdot \prod_{i=1}^{n} P(x_i \mid C)$$
""")
    with ct2:
        caixa("<b>Por que 'ingênuo'?</b><br>Porque na prática as features raramente são "
              "independentes. Ex: altura e peso de uma pessoa são correlacionadas. "
              "Mesmo assim o NB funciona porque o que importa é o <b>ranking relativo</b> "
              "das probabilidades, não os valores absolutos.",
              cor="#fff8e1", borda="#f59e0b")
        caixa("<b>GaussianNB — distribuição normal</b><br>Assume que cada feature segue uma "
              "<b>distribuição gaussiana</b> dentro de cada classe. Estima média (μ) e "
              "variância (σ²) por feature × classe usando os dados de treino.")

    divider()
    secao("Distribuição Gaussiana — Visualização Interativa")
    cg1, cg2 = st.columns([1, 1])
    with cg1:
        mu    = st.slider("Média (μ)", -5.0, 5.0, 0.0, 0.1, key="mu")
        sigma = st.slider("Desvio (σ)", 0.5, 4.0, 1.0, 0.1, key="sig")
        xval  = st.slider("Valor x:", -10.0, 10.0, 1.0, 0.1, key="xv")
    with cg2:
        p_x_c = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((xval-mu)/sigma)**2)
        st.metric("P(x | C)", f"{p_x_c:.6f}")
        st.metric("μ", f"{mu:.1f}"); st.metric("σ", f"{sigma:.1f}")

    x_range = np.linspace(-10, 10, 500)
    gauss   = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x_range-mu)/sigma)**2)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x_range, gauss, color="#667eea", linewidth=2.5, label=f"N(μ={mu}, σ={sigma})")
    ax.axvline(xval, color="gold", linewidth=2, label=f"x={xval:.1f}")
    ax.scatter([xval], [p_x_c], color="gold", s=200, zorder=5,
               label=f"P(x|C)={p_x_c:.6f}")
    ax.fill_between(x_range, gauss,
                    where=(x_range >= mu-sigma) & (x_range <= mu+sigma),
                    alpha=0.2, color="#667eea", label="±1σ (68%)")
    ax.set_xlabel("x"); ax.set_ylabel("Densidade")
    ax.set_title("GaussianNB — Distribuição Normal", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    save_and_show(fig, "gnb_gaussiana")

    divider()
    secao("Variações do Naive Bayes")
    cv1, cv2, cv3 = st.columns(3)
    with cv1:
        caixa("<b>GaussianNB</b><br>Para features <b>contínuas</b>. Assume distribuição "
              "normal. Ideal quando as features numéricas seguem (aproximadamente) uma "
              "distribuição gaussiana em cada classe.")
    with cv2:
        caixa("<b>MultinomialNB</b><br>Para features <b>discretas / contagens</b>. "
              "Muito usado em <b>classificação de texto</b> (frequência de palavras, "
              "bag-of-words com contagens, filtragem de spam).")
    with cv3:
        caixa("<b>BernoulliNB</b><br>Para features <b>binárias</b> (0/1). Indica "
              "presença ou ausência de uma característica. Usado em texto com "
              "vetores bag-of-words binários.")

    divider()
    caixa("<b>⚠️ GaussianNB e OneHotEncoder não combinam bem</b><br>"
          "O GaussianNB assume distribuição normal contínua. OneHotEncoder cria colunas "
          "binárias (0/1) que violam essa suposição. Solução: use "
          "<code>remainder='drop'</code> no ColumnTransformer para excluir as features "
          "categóricas ao usar GNB.", cor="#fff0f0", borda="#e53e3e")

    divider()
    with st.expander("Ver código Python — GaussianNB"):
        code_block("""
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score

# Apenas features numéricas (SEM OneHotEncoder para GNB)
pre_gnb = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
    ],
    remainder='drop'   # descarta categóricas
)

pipe_gnb = Pipeline(steps=[
    ('pre', pre_gnb),
    ('gnb', GaussianNB())
])

pipe_gnb.fit(X_train, y_train)
y_pred = pipe_gnb.predict(X_test)
probas  = pipe_gnb.predict_proba(X_test)   # probabilidade por classe

# Validação cruzada
cv_acc = cross_val_score(pipe_gnb, X, y, cv=10, scoring='accuracy')
cv_f1  = cross_val_score(pipe_gnb, X, y, cv=10, scoring='f1_macro')
print(f"Accuracy: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
""")

# ============================================================================
# NAIVE BAYES — PINGUINS
# ============================================================================
elif pagina == "🐧  Naive Bayes — Pinguins":
    st.markdown('<h2>🐧 GaussianNB — Dataset Penguins</h2>', unsafe_allow_html=True)
    divider()

    caixa("<b>Palmer Penguins com GaussianNB</b><br>Usamos as <b>4 features numéricas</b> "
          "(comprimento/profundidade do bico, nadadeira, massa corporal). As features "
          "categóricas (ilha, sexo) são descartadas — GNB só funciona bem com features "
          "que seguem distribuição aproximadamente normal.")

    with st.spinner("Carregando Penguins..."):
        df_pen = carregar_penguins()
    st.success(f"Dataset: {df_pen.shape[0]} pinguins × {df_pen.shape[1]} colunas")

    secao("Análise Exploratória")
    cp1, cp2 = st.columns(2)
    with cp1:
        st.markdown("**Distribuição por espécie:**")
        st.dataframe(df_pen["espécie"].value_counts().reset_index(), use_container_width=True)
    with cp2:
        st.markdown("**Primeiras linhas:**")
        st.dataframe(df_pen.head(6), use_container_width=True)

    divider()
    num_cols = ["comprimento_bico_mm","profundidade_bico_mm",
                "comprimento_nadadeira_mm","massa_corporal_g"]
    paleta   = {"Adelie":"#4C72B0","Chinstrap":"#DD8452","Gentoo":"#55A868"}

    secao("Visualizações")
    cv1, cv2 = st.columns(2)
    with cv1:
        st.markdown("**Scatter: Bico (comprimento × profundidade)**")
        fig, ax = plt.subplots(figsize=(6, 4))
        for esp, cor in paleta.items():
            sub = df_pen[df_pen["espécie"]==esp]
            ax.scatter(sub["comprimento_bico_mm"], sub["profundidade_bico_mm"],
                       color=cor, label=esp, alpha=0.65, s=30)
        ax.set_xlabel("Comprimento bico (mm)"); ax.set_ylabel("Profundidade bico (mm)")
        ax.set_title("Bico por Espécie"); ax.legend(); ax.grid(alpha=0.3)
        save_and_show(fig, "pen_gnb_scatter")
    with cv2:
        st.markdown("**Histogramas das 4 features numéricas**")
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        axes = axes.flatten()
        for i, col in enumerate(num_cols):
            for esp, cor in paleta.items():
                axes[i].hist(df_pen[df_pen["espécie"]==esp][col], bins=20,
                             alpha=0.5, color=cor, label=esp)
            axes[i].set_title(col, fontsize=9); axes[i].grid(alpha=0.3)
        axes[0].legend(fontsize=7); plt.tight_layout()
        save_and_show(fig, "pen_gnb_histogramas")

    divider()
    secao("Pipeline: StandardScaler + GaussianNB (só numéricas)")

    test_size = st.slider("Proporção do teste:", 0.1, 0.4, 0.2, 0.05, key="ts_gnb")

    X_pen = df_pen.drop("espécie", axis=1); y_pen = df_pen["espécie"]
    pre_gnb = ColumnTransformer(
        transformers=[("num", StandardScaler(), num_cols)], remainder="drop")
    pipe_gnb = Pipeline([("pre", pre_gnb), ("gnb", GaussianNB())])
    Xtr, Xte, ytr, yte = train_test_split(X_pen, y_pen, test_size=test_size,
                                           stratify=y_pen, random_state=42)
    pipe_gnb.fit(Xtr, ytr)
    ypred = pipe_gnb.predict(Xte)
    acc = accuracy_score(yte, ypred); f1 = f1_score(yte, ypred, average="macro")

    cm1, cm2, cm3 = st.columns(3)
    cm1.metric("Accuracy", f"{acc:.4f}"); cm2.metric("F1-macro", f"{f1:.4f}")
    cm3.metric("Amostras teste", len(yte))

    secao("Relatório de Classificação")
    st.dataframe(pd.DataFrame(classification_report(yte, ypred, output_dict=True)).T.round(4),
                 use_container_width=True)

    secao("Matriz de Confusão")
    classes = sorted(y_pen.unique())
    cm_arr = confusion_matrix(yte, ypred, labels=classes)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=classes, yticklabels=classes, linewidths=1)
    ax.set_xlabel("Previsto"); ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão — GNB (Penguins)")
    save_and_show(fig, "pen_gnb_cm")

    divider()
    secao("Validação Cruzada cv=10")
    with st.spinner("Executando CV..."):
        cv_acc = cross_val_score(pipe_gnb, X_pen, y_pen, cv=10, scoring="accuracy")
        cv_f1  = cross_val_score(pipe_gnb, X_pen, y_pen, cv=10, scoring="f1_macro")
    cva, cvf = st.columns(2)
    cva.metric("CV Accuracy", f"{cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
    cvf.metric("CV F1-macro", f"{cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

    fig, ax = plt.subplots(figsize=(9, 4))
    dobras = range(1, 11)
    ax.bar([d-.2 for d in dobras], cv_acc, width=.35, label="Accuracy",
           color="#667eea", alpha=.85)
    ax.bar([d+.2 for d in dobras], cv_f1,  width=.35, label="F1-macro",
           color="#764ba2", alpha=.85)
    ax.axhline(cv_acc.mean(), color="#667eea", linestyle="--", linewidth=1.5, alpha=.7)
    ax.axhline(cv_f1.mean(),  color="#764ba2", linestyle="--", linewidth=1.5, alpha=.7)
    ax.set_xlabel("Dobra"); ax.set_ylabel("Métrica")
    ax.set_title("CV cv=10 — GaussianNB (Penguins)")
    ax.set_xticks(list(dobras)); ax.legend(); ax.grid(alpha=.3, axis="y")
    save_and_show(fig, "pen_gnb_cv")

    divider()
    with st.expander("Ver código Python — GNB Penguins"):
        code_block("""
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

num_cols = ["comprimento_bico_mm","profundidade_bico_mm",
            "comprimento_nadadeira_mm","massa_corporal_g"]

# Apenas numéricas — categóricas violam suposição gaussiana
pre = ColumnTransformer(
    transformers=[("num", StandardScaler(), num_cols)],
    remainder="drop"
)
pipe = Pipeline([("pre", pre), ("gnb", GaussianNB())])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
pipe.fit(X_train, y_train)
print(classification_report(y_test, pipe.predict(X_test)))

cv_acc = cross_val_score(pipe, X, y, cv=10, scoring="accuracy")
print(f"Accuracy: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
""")

# ============================================================================
# SVM — TEORIA E KERNELS
# ============================================================================
elif pagina == "⚔️  SVM — Teoria e Kernels":
    st.markdown('<h2>⚔️ SVM — Support Vector Machine</h2>', unsafe_allow_html=True)
    divider()

    caixa("<b>O que é SVM?</b><br>Classificador <b>baseado em margem</b>. Busca o "
          "<b>hiperplano ótimo</b> que separa as classes com a <b>maior margem possível</b>. "
          "Os pontos mais próximos da fronteira — os <b>vetores de suporte</b> — definem "
          "completamente a posição do hiperplano. Suporta dados não lineares via "
          "<b>Kernel Trick</b>.")

    divider()
    secao("Conceitos Fundamentais")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        caixa("<b>Hiperplano e Margem</b><br>"
              "Em 2D → reta &nbsp;|&nbsp; Em 3D → plano &nbsp;|&nbsp; Em nD → hiperplano.<br><br>"
              "A <b>margem</b> é a distância entre o hiperplano e os vetores de suporte. "
              "O SVM <b>maximiza</b> essa margem para generalizar melhor.")
    with cc2:
        caixa("<b>Parâmetro C</b><br>"
              "• <b>C pequeno</b>: mais tolerância a erros → margem <b>larga</b> "
              "→ modelo mais simples<br>"
              "• <b>C grande</b>: pouca tolerância → margem <b>estreita</b> "
              "→ risco de overfitting",
              cor="#fff8e1", borda="#f59e0b")
    with cc3:
        caixa("<b>Kernel Trick</b><br>"
              "Transforma os dados em dimensão superior sem calcular explicitamente:<br>"
              "• <b>linear</b>: fronteira reta<br>"
              "• <b>RBF</b>: fronteiras curvas (padrão)<br>"
              "• <b>poly</b>: fronteiras polinomiais")

    divider()
    secao("Demonstração Interativa — Fronteira de Decisão")
    cd1, cd2 = st.columns(2)
    with cd1:
        C_d     = st.select_slider("C:", [0.01,0.1,1.0,10.0,100.0], value=1.0, key="C_s")
        gamma_d = st.select_slider("Gamma (RBF):", [0.01,0.1,0.5,1.0,3.0,10.0], value=0.5, key="g_s")
    with cd2:
        kernel_d = st.selectbox("Kernel:", ["linear","rbf","poly"], key="ks")
        ds_type  = st.selectbox("Dataset:", ["Quase Linear (blobs)","Não Linear (moons)"], key="ds")

    np.random.seed(42)
    if ds_type == "Quase Linear (blobs)":
        Xd, yd = make_blobs(n_samples=120, centers=2, cluster_std=1.3, random_state=42)
    else:
        Xd, yd = make_moons(n_samples=180, noise=0.20, random_state=42)

    pipe_d = Pipeline([("scl",StandardScaler()),("svc",SVC(kernel=kernel_d,C=C_d,gamma=gamma_d))])
    pipe_d.fit(Xd, yd)
    h = 0.05
    x_min,x_max = Xd[:,0].min()-1, Xd[:,0].max()+1
    y_mn, y_mx  = Xd[:,1].min()-1, Xd[:,1].max()+1
    xx,yy = np.meshgrid(np.arange(x_min,x_max,h), np.arange(y_mn,y_mx,h))
    Z = pipe_d.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
    acc_d = accuracy_score(yd, pipe_d.predict(Xd))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.contourf(xx, yy, Z, alpha=0.2, cmap="RdBu")
    ax.scatter(Xd[yd==0,0], Xd[yd==0,1], color="#4C72B0", s=35, alpha=.75, label="Classe 0")
    ax.scatter(Xd[yd==1,0], Xd[yd==1,1], color="#DD8452", s=35, alpha=.75, label="Classe 1")
    ax.set_title(f"SVM kernel={kernel_d} | C={C_d} | γ={gamma_d} | Acc={acc_d:.3f}")
    ax.legend(); ax.grid(alpha=.25)
    save_and_show(fig, f"svm_demo_{kernel_d}")

    divider()
    secao("Comparação: C pequeno vs C grande")
    np.random.seed(42)
    Xb, yb = make_blobs(n_samples=100, centers=2, cluster_std=1.4, random_state=42)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axi, Cv, titulo in [(axes[0],0.01,"C=0.01 → Margem larga"),(axes[1],100,"C=100 → Margem estreita")]:
        clf = Pipeline([("scl",StandardScaler()),("svc",SVC(kernel="linear",C=Cv))])
        clf.fit(Xb, yb)
        xm,xM = Xb[:,0].min()-1, Xb[:,0].max()+1
        ym,yM = Xb[:,1].min()-1, Xb[:,1].max()+1
        xx2,yy2 = np.meshgrid(np.arange(xm,xM,.05), np.arange(ym,yM,.05))
        Z2 = clf.predict(np.c_[xx2.ravel(),yy2.ravel()]).reshape(xx2.shape)
        axi.contourf(xx2, yy2, Z2, alpha=.15, cmap="RdBu")
        axi.scatter(Xb[yb==0,0], Xb[yb==0,1], color="#4C72B0", s=35, alpha=.75)
        axi.scatter(Xb[yb==1,0], Xb[yb==1,1], color="#DD8452", s=35, alpha=.75)
        axi.set_title(titulo); axi.grid(alpha=.25)
    plt.tight_layout()
    save_and_show(fig, "svm_C_comp")

    divider()
    secao("Gamma (γ) — Alcance do Kernel RBF")
    np.random.seed(42)
    Xm, ym = make_moons(n_samples=200, noise=0.15, random_state=42)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axi, gv, titulo in [
        (axes[0], 0.1,  "γ=0.1 → Suave (underfitting)"),
        (axes[1], 1.0,  "γ=1.0 → Equilibrado"),
        (axes[2], 10.0, "γ=10.0 → Ondulado (overfitting)"),
    ]:
        clf = Pipeline([("scl",StandardScaler()),("svc",SVC(kernel="rbf",C=1.0,gamma=gv))])
        clf.fit(Xm, ym)
        xm2,xM2 = Xm[:,0].min()-.5, Xm[:,0].max()+.5
        ym2,yM2 = Xm[:,1].min()-.5, Xm[:,1].max()+.5
        xx3,yy3 = np.meshgrid(np.arange(xm2,xM2,.05), np.arange(ym2,yM2,.05))
        Z3 = clf.predict(np.c_[xx3.ravel(),yy3.ravel()]).reshape(xx3.shape)
        axi.contourf(xx3, yy3, Z3, alpha=.2, cmap="RdBu")
        axi.scatter(Xm[ym==0,0], Xm[ym==0,1], color="#4C72B0", s=20, alpha=.7)
        axi.scatter(Xm[ym==1,0], Xm[ym==1,1], color="#DD8452", s=20, alpha=.7)
        axi.set_title(titulo, fontsize=10); axi.grid(alpha=.2)
    plt.tight_layout()
    save_and_show(fig, "svm_gamma_comp")

    divider()
    with st.expander("Ver código Python — SVM"):
        code_block("""
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Kernel RBF (padrão, bom ponto de partida)
pipe_rbf = Pipeline([
    ("scl", StandardScaler()),
    ("svc", SVC(kernel="rbf", C=1.0, gamma="scale"))
])
pipe_rbf.fit(X_train, y_train)
y_pred = pipe_rbf.predict(X_test)

# Kernel linear
pipe_linear = Pipeline([
    ("scl", StandardScaler()),
    ("svc", SVC(kernel="linear", C=1.0))
])

# Para obter probabilidades
pipe_prob = Pipeline([
    ("scl", StandardScaler()),
    ("svc", SVC(kernel="rbf", probability=True))
])
probas = pipe_prob.predict_proba(X_test)
""")

# ============================================================================
# SVM — PINGUINS + GRIDSEARCH
# ============================================================================
elif pagina == "🐧  SVM — Pinguins + GridSearch":
    st.markdown('<h2>🐧 SVM — Penguins + GridSearchCV</h2>', unsafe_allow_html=True)
    divider()

    caixa("<b>GridSearchCV com Pipeline</b><br>"
          "Testamos todas as combinações do <code>param_grid</code> usando "
          "<code>StratifiedKFold(n_splits=5)</code>. O prefixo dos parâmetros deve "
          "ser o <b>nome do passo no Pipeline</b> + <code>__</code>. Ex: passo chamado "
          "<code>'svm'</code> → parâmetro kernel vira <code>svm__kernel</code>.")

    with st.spinner("Carregando Penguins..."):
        df_pen = carregar_penguins()

    X_pen = df_pen.drop("espécie", axis=1); y_pen = df_pen["espécie"]
    num_c = ["comprimento_bico_mm","profundidade_bico_mm",
             "comprimento_nadadeira_mm","massa_corporal_g"]
    cat_c = ["ilha","sexo"]

    Xtr, Xte, ytr, yte = train_test_split(X_pen, y_pen, test_size=.3,
                                           stratify=y_pen, random_state=42)
    pre_svm = ColumnTransformer([
        ("num", StandardScaler(), num_c),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_c),
    ])
    pipe_svc = Pipeline([("pre",pre_svm),("svm",SVC(kernel="linear",random_state=42))])
    pipe_svc.fit(Xtr, ytr); ypred = pipe_svc.predict(Xte)
    acc_base = accuracy_score(yte, ypred); f1_base = f1_score(yte, ypred, average="macro")

    secao("Baseline — SVC linear (parâmetros padrão)")
    cb1, cb2, cb3 = st.columns(3)
    cb1.metric("Accuracy", f"{acc_base:.4f}")
    cb2.metric("F1-macro", f"{f1_base:.4f}")
    cb3.metric("Amostras teste", len(yte))

    secao("Relatório")
    st.dataframe(pd.DataFrame(classification_report(yte,ypred,output_dict=True)).T.round(4),
                 use_container_width=True)

    secao("Matriz de Confusão")
    classes_p = sorted(y_pen.unique())
    cm_arr = confusion_matrix(yte, ypred, labels=classes_p)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=classes_p, yticklabels=classes_p, linewidths=1)
    ax.set_xlabel("Previsto"); ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão — SVC Baseline (Penguins)")
    save_and_show(fig, "pen_svm_cm_baseline")

    divider()
    secao("GridSearchCV — Otimização de Hiperparâmetros")

    caixa("<b>Dois subgrids na lista param_grid:</b><br>"
          "• <code>linear</code>: só precisa de C (sem gamma)<br>"
          "• <code>rbf</code>: precisa de C e gamma<br><br>"
          "Ao passar uma <b>lista de dicionários</b>, o GridSearch testa cada "
          "subgrid separadamente — evita combinações inválidas.",
          cor="#fff8e1", borda="#f59e0b")

    with st.spinner("GridSearchCV em execução (~15s)..."):
        param_grid_gs = [
            {"svm__kernel": ["linear"], "svm__C": [0.1, 1, 10]},
            {"svm__kernel": ["rbf"],    "svm__C": [0.1, 1, 10],
             "svm__gamma": ["scale", 0.1, 1.0]},
        ]
        cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        gs = GridSearchCV(pipe_svc, param_grid_gs, scoring="accuracy", cv=cv_strat)
        gs.fit(Xtr, ytr)

    caixa(f"<b>✅ Melhores parâmetros:</b> {gs.best_params_}<br>"
          f"<b>Melhor Accuracy (CV=5):</b> {gs.best_score_:.4f}",
          cor="#f0fff4", borda="#38a169")

    secao("Tabela completa da busca")
    cols_res = ["param_svm__kernel","param_svm__C","param_svm__gamma",
                "mean_test_score","rank_test_score"]
    df_gs = pd.DataFrame(gs.cv_results_)[cols_res].sort_values("rank_test_score")
    st.dataframe(df_gs.reset_index(drop=True).round(4), use_container_width=True)

    secao("Avaliação do melhor modelo no conjunto de teste")
    best = gs.best_estimator_; ypred_best = best.predict(Xte)
    acc_best = accuracy_score(yte, ypred_best); f1_best = f1_score(yte, ypred_best, average="macro")
    gb1, gb2 = st.columns(2)
    gb1.metric("Accuracy (teste)", f"{acc_best:.4f}", delta=f"{acc_best-acc_base:+.4f}")
    gb2.metric("F1-macro (teste)", f"{f1_best:.4f}",  delta=f"{f1_best-f1_base:+.4f}")

    divider()
    with st.expander("Ver código Python — SVM + GridSearchCV"):
        code_block("""
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

# Dois subgrids: linear (sem gamma) e rbf (com gamma)
param_grid = [
    {"svm__kernel": ["linear"], "svm__C": [0.1, 1, 10]},
    {"svm__kernel": ["rbf"],    "svm__C": [0.1, 1, 10],
     "svm__gamma":  ["scale", 0.1, 1.0]},
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

gs = GridSearchCV(
    estimator=pipe_svc,   # Pipeline com passo chamado "svm"
    param_grid=param_grid,
    scoring="accuracy",
    cv=cv,
)
gs.fit(X_train, y_train)

print("Melhores parâmetros:", gs.best_params_)
print(f"Melhor accuracy (CV=5): {gs.best_score_:.4f}")

# Melhor modelo treinado
best = gs.best_estimator_
y_pred = best.predict(X_test)
print(classification_report(y_test, y_pred))
""")

# ============================================================================
# EXERCÍCIO — WINE
# ============================================================================
elif pagina == "🍷  Exercício — Dataset Wine":
    st.markdown('<h2>🍷 Exercício — Dataset Wine</h2>', unsafe_allow_html=True)
    divider()

    caixa("<b>Dataset Wine (sklearn / UCI)</b><br>178 amostras de vinho de <b>3 tipos</b> "
          "com 13 características químicas (álcool, ácido málico, magnésio, flavonóides, "
          "etc.). Problema multi-classe com apenas features numéricas — ideal para comparar "
          "KNN, GNB e SVM com GridSearchCV.")

    with st.spinner("Carregando Wine..."):
        df_wine, target_names = carregar_wine()
    st.success(f"Dataset: {df_wine.shape[0]} amostras × {df_wine.shape[1]} colunas")

    secao("Análise Exploratória")
    cw1, cw2 = st.columns(2)
    with cw1:
        st.markdown("**Distribuição do target:**")
        vc = df_wine["target"].value_counts().sort_index()
        vc.index = [f"Tipo {i} ({target_names[i]})" for i in vc.index]
        st.dataframe(vc.reset_index(), use_container_width=True)
    with cw2:
        st.markdown("**Estatísticas descritivas:**")
        st.dataframe(df_wine.describe().round(3), use_container_width=True)

    divider()
    secao("Heatmap de Correlação")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df_wine.drop("target",axis=1).corr(), annot=True, fmt=".2f",
                cmap="coolwarm", center=0, ax=ax, linewidths=.3, annot_kws={"size":7})
    ax.set_title("Correlação — Wine")
    save_and_show(fig, "wine_correlacao")

    divider()
    secao("Comparativo: KNN vs GNB vs SVM (GridSearch + CV=10)")

    X_wine = df_wine.drop("target", axis=1); y_wine = df_wine["target"]
    pre_wine = ColumnTransformer([("num", StandardScaler(), X_wine.columns.tolist())])
    cv_w = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    modelos_config = [
        ("KNN",         KNeighborsClassifier(),
         {"clf__n_neighbors":[3,5,7,9], "clf__weights":["uniform","distance"],
          "clf__metric":["euclidean","manhattan"]}),
        ("Naive Bayes", GaussianNB(),
         {"clf__var_smoothing":[1e-9,1e-8,1e-7,1e-6,1e-5]}),
        ("SVM",         SVC(),
         [{"clf__kernel":["linear"],"clf__C":[0.1,1,10]},
          {"clf__kernel":["rbf"],"clf__C":[0.1,1,10],"clf__gamma":["scale",0.1,1.0]}]),
    ]

    with st.spinner("Treinando 3 modelos com GridSearch e CV=10 (~20s)..."):
        resultados = []
        for nome, estimator, pgrid in modelos_config:
            pipe_w = Pipeline([("pre",pre_wine),("clf",estimator)])
            gs_w = GridSearchCV(pipe_w, pgrid, scoring="accuracy", cv=cv_w)
            gs_w.fit(X_wine, y_wine)
            cv_acc = cross_val_score(gs_w.best_estimator_, X_wine, y_wine,
                                     cv=10, scoring="accuracy")
            cv_f1  = cross_val_score(gs_w.best_estimator_, X_wine, y_wine,
                                     cv=10, scoring="f1_macro")
            resultados.append({
                "Modelo": nome,
                "Melhores params": str(gs_w.best_params_),
                "Acc CV=10":  f"{cv_acc.mean():.4f} ± {cv_acc.std():.4f}",
                "F1 CV=10":   f"{cv_f1.mean():.4f} ± {cv_f1.std():.4f}",
            })

    st.dataframe(pd.DataFrame(resultados), use_container_width=True)

    nomes = [r["Modelo"] for r in resultados]
    accs  = [float(r["Acc CV=10"].split(" ±")[0]) for r in resultados]
    f1s   = [float(r["F1 CV=10"].split(" ±")[0]) for r in resultados]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(nomes))
    ax.bar(x-.2, accs, width=.35, label="Accuracy", color="#667eea", alpha=.9)
    ax.bar(x+.2, f1s,  width=.35, label="F1-macro",  color="#764ba2", alpha=.9)
    for i,(a,f) in enumerate(zip(accs,f1s)):
        ax.text(i-.2, a+.005, f"{a:.3f}", ha="center", fontsize=9)
        ax.text(i+.2, f+.005, f"{f:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(nomes, fontsize=11)
    ax.set_ylabel("Métrica (CV=10)")
    ax.set_title("Comparativo — KNN vs GNB vs SVM | Dataset Wine")
    ax.set_ylim(0.85, 1.05); ax.legend(); ax.grid(alpha=.3, axis="y")
    save_and_show(fig, "wine_comparativo")

    divider()
    with st.expander("Ver código Python — Exercício Wine completo"):
        code_block("""
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np

data = load_wine()
df   = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

X = df.drop("target", axis=1)
y = df["target"]

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

pre = ColumnTransformer([("num", StandardScaler(), X.columns.tolist())])
cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

from sklearn.svm import SVC
pipe_svc = Pipeline([("pre", pre), ("clf", SVC())])
param_grid = [
    {"clf__kernel": ["linear"], "clf__C": [0.1, 1, 10]},
    {"clf__kernel": ["rbf"],    "clf__C": [0.1, 1, 10],
     "clf__gamma":  ["scale", 0.1, 1.0]},
]
gs = GridSearchCV(pipe_svc, param_grid, scoring="accuracy", cv=cv)
gs.fit(X, y)

print("Melhores parâmetros:", gs.best_params_)
cv_acc = cross_val_score(gs.best_estimator_, X, y, cv=10, scoring="accuracy")
print(f"Accuracy CV=10: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
""")

    divider()
    st.markdown("""<div class="footer">
        <p style="margin:0;font-size:1.1rem;font-weight:700;">Machine Learning — Aula 05</p>
        <p style="margin:.3rem 0 0;font-size:.9rem;opacity:.8;">
        Naive Bayes e SVM · Cláudio Ferreira Neves · SENAI/SC</p></div>""",
        unsafe_allow_html=True)
