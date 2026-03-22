"""
=============================================================================
Machine Learning — Aula 08 / Comparação de Modelos — Alzheimer
Aplicação Streamlit interativa — material didático para iniciantes

Autor       : Cláudio Ferreira Neves
Cargo atual : Analista de BI — Save Co. / Especialista de Ensino II — SENAI/SC
Certificação: DATA ANALYST CERTIFIED PROFESSIONAL © (DACP)
=============================================================================
"""

import os
import numpy as np
import pandas as pd

PAGE_PORTAL  = "pages/Portal.py"
PAGE_AULA_07 = "pages/Aula_07.py"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    train_test_split, cross_validate, cross_val_score,
    GridSearchCV, StratifiedKFold
)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, make_scorer
)

try:
    from xgboost import XGBClassifier
    XGBOOST_OK = True
except ImportError:
    XGBOOST_OK = False

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

URL_ALZHEIMER = (
    "https://raw.githubusercontent.com/guilhermebernieri/"
    "pos_graduacao/refs/heads/main/alzheimers_disease_data.csv"
)


# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def save_and_show(fig, filename):
    path = os.path.join(OUTPUTS_DIR, f"{filename}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    st.pyplot(fig)
    plt.close(fig)

def caixa_conceito(html):
    st.markdown(f'<div class="concept-box">{html}</div>', unsafe_allow_html=True)

def caixa_dica(html):
    st.markdown(f'<div class="tip-box">💡 {html}</div>', unsafe_allow_html=True)

def caixa_atencao(html):
    st.markdown(f'<div class="warn-box">⚠️ {html}</div>', unsafe_allow_html=True)

def caixa_sucesso(html):
    st.markdown(f'<div class="explain-box">{html}</div>', unsafe_allow_html=True)

def secao(texto):
    st.markdown(f"### {texto}")

def divider():
    st.markdown("---")

def code_block(code, title=""):
    if title:
        st.markdown(f"**{title}**")
    st.code(code, language="python")


# ============================================================================
# DADOS
# ============================================================================

@st.cache_data
def carregar_alzheimer():
    df = pd.read_csv(URL_ALZHEIMER)
    df.drop(columns=["PatientID", "DoctorInCharge"], inplace=True, errors="ignore")
    return df


# ============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="Comparação de Modelos — Alzheimer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
        <h2 style="margin:0; font-size:1.3rem;">🧬 Aula 08</h2>
        <p style="margin:0.4rem 0 0; font-size:0.8rem; opacity:0.8;">Comparação de Modelos</p>
    </div>
    """, unsafe_allow_html=True)

    pagina = st.radio(
        "Navegação",
        options=[
            "🏠  Início",
            "🔬  Dataset Alzheimer — EDA",
            "⚙️  Pré-processamento",
            "🤖  Treinamento dos Modelos",
            "🏆  Comparativo Final",
            "🎯  Ensemble — VotingClassifier",
            "💾  Salvar e Carregar Modelo",
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
# AUTOR E NAVEGAÇÃO
# ============================================================================

# ── navegação entre aulas ────────────────────────────────────────────────────
from nav import tab_nav
tab_nav(8)  # replace CURRENT_NUMBER with the correct int

st.markdown(
    "<p style='text-align:center; font-size:0.95rem; color:#667eea; "
    "font-weight:600; margin-bottom:0;'>Autor: Especialista Cláudio Ferreira Neves</p>",
    unsafe_allow_html=True,
)
_nav_l, _nav_m, _nav_r = st.columns([1.2, 4, 1.2])
with _nav_l:
    if st.button("← Aula 07", use_container_width=True, key="nav_prev"):
        st.switch_page(PAGE_AULA_07)
with _nav_m:
    if st.button("🏠 Portal", use_container_width=True, key="nav_portal"):
        st.switch_page(PAGE_PORTAL)
with _nav_r:
    st.markdown("")  # última aula — sem próxima


# ============================================================================
# PÁGINA: INÍCIO
# ============================================================================

if pagina == "🏠  Início":

    st.markdown("""
    <div style='text-align:center; padding: 2rem 0 1rem;'>
        <h1 style='font-size:2.8rem; font-weight:800; color:#1a1a2e; margin-bottom:0.3rem;'>
            🧬 Comparação de Modelos
        </h1>
        <p style='font-size:1.15rem; color:#555; max-width:760px; margin:0 auto;'>
            Aula final: aplique todos os modelos estudados no diagnóstico de Alzheimer,
            compare resultados com rigor e construa um ensemble dos melhores modelos.
        </p>
    </div>
    """, unsafe_allow_html=True)

    divider()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><h2>6+</h2><p>Modelos</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h2>2149</h2><p>Pacientes</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h2>🧬</h2><p>Alzheimer</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><h2>🏆</h2><p>Ensemble</p></div>', unsafe_allow_html=True)

    divider()
    st.markdown("## 🎯 Problema de Negócio")
    caixa_conceito("""
    <b>Diagnóstico de Alzheimer com Machine Learning</b><br><br>
    Você foi contratado como cientista de dados para analisar um dataset com informações
    clínicas, demográficas, comportamentais e cognitivas de <b>2.149 pacientes</b> e
    construir um modelo capaz de prever se um paciente <b>tem ou não o diagnóstico de Alzheimer</b>.<br><br>
    O desafio final é comparar todos os algoritmos estudados ao longo do curso
    e escolher o melhor — ou combiná-los em um <b>ensemble</b> para maximizar a precisão.
    """)

    divider()
    st.markdown("## 📋 Metodologia CRISP-DM")
    passos = [
        ("1. Entendimento do Negócio", "Qual pergunta queremos responder? Prever diagnóstico de Alzheimer."),
        ("2. Entendimento dos Dados", "EDA: distribuições, correlações, valores ausentes, outliers."),
        ("3. Preparação dos Dados", "Pré-processamento: encoding, escalonamento, split treino/teste."),
        ("4. Modelagem", "Treinar KNN, SVM, DT, RF, XGBoost, LogReg com cross-validation."),
        ("5. Avaliação", "Comparar Accuracy e F1-macro entre todos os modelos."),
        ("6. Deploy", "Salvar o melhor modelo com joblib para uso em produção."),
    ]
    for num, (titulo, desc) in enumerate(passos, 1):
        st.markdown(
            f"<div style='background:#f8fafc; border-left:4px solid #667eea; "
            f"border-radius:0 10px 10px 0; padding:0.7rem 1.2rem; margin-bottom:0.4rem;'>"
            f"<b style='color:#667eea;'>{titulo}</b><br>"
            f"<span style='color:#555; font-size:0.88rem;'>{desc}</span></div>",
            unsafe_allow_html=True,
        )

    divider()
    st.markdown("""
    <div class="footer">
        <p style="margin:0; font-size:1.1rem; font-weight:700;">Machine Learning — Aula 08</p>
        <p style="margin:0.3rem 0 0; font-size:0.9rem; opacity:0.8;">
            Comparação de Modelos — Alzheimer &nbsp;·&nbsp;
            Cláudio Ferreira Neves &nbsp;·&nbsp; SENAI/SC
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# PÁGINA: EDA
# ============================================================================

elif pagina == "🔬  Dataset Alzheimer — EDA":

    st.markdown('<p class="section-title">🔬 Dataset Alzheimer — Análise Exploratória</p>',
                unsafe_allow_html=True)
    divider()

    caixa_conceito("""
    <b>Dataset: Alzheimer's Disease Dataset</b><br>
    2.149 pacientes com informações clínicas, demográficas e cognitivas.
    Target: <code>Diagnosis</code> (0 = não diagnosticado, 1 = diagnosticado).
    A coluna <code>DoctorInCharge</code> foi removida (dados confidenciais).
    """)

    with st.spinner("Carregando dataset Alzheimer..."):
        try:
            df_alz = carregar_alzheimer()
            st.success(f"Dataset carregado: {df_alz.shape[0]} linhas × {df_alz.shape[1]} colunas")
        except Exception as e:
            st.error(f"Erro ao carregar: {e}")
            st.stop()

    secao("Visão Geral")
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.dataframe(df_alz.head(8), use_container_width=True)
    with col_e2:
        st.markdown("**Info do dataset:**")
        info_dict = {
            "Linhas": df_alz.shape[0],
            "Colunas": df_alz.shape[1],
            "Valores nulos": int(df_alz.isnull().sum().sum()),
            "Duplicatas": int(df_alz.duplicated().sum()),
        }
        st.dataframe(pd.DataFrame(info_dict.items(), columns=["Métrica", "Valor"]),
                     use_container_width=True)

        vc_alz = df_alz["Diagnosis"].value_counts().reset_index()
        vc_alz.columns = ["Diagnóstico", "Contagem"]
        vc_alz["Diagnóstico"] = vc_alz["Diagnóstico"].map({0: "Não diagnosticado", 1: "Diagnosticado"})
        st.dataframe(vc_alz, use_container_width=True)

    divider()
    secao("Identificação de Features")

    numerical_alz = [col for col in df_alz.columns if df_alz[col].nunique() > 10 and col != "Diagnosis"]
    categorical_alz = [col for col in df_alz.columns
                       if df_alz[col].nunique() <= 10 and col != "Diagnosis"]

    col_n, col_c = st.columns(2)
    with col_n:
        st.markdown(f"**Features numéricas ({len(numerical_alz)}):**")
        st.write(numerical_alz)
    with col_c:
        st.markdown(f"**Features categóricas/binárias ({len(categorical_alz)}):**")
        st.write(categorical_alz)

    divider()
    secao("Distribuição do Target vs Features Numéricas (top 6)")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, feat in enumerate(numerical_alz[:6]):
        for diag, cor, label in [(0, "#667eea", "Não diagnosticado"), (1, "#e53e3e", "Diagnosticado")]:
            subset = df_alz[df_alz["Diagnosis"] == diag][feat]
            axes[i].hist(subset, bins=25, alpha=0.55, color=cor, label=label, density=True)
        axes[i].set_title(feat, fontsize=9)
        axes[i].grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    plt.suptitle("Distribuição das Features por Diagnóstico", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_and_show(fig, "alz_histogramas_target")

    divider()
    secao("Heatmap de Correlação (features numéricas)")
    fig, ax = plt.subplots(figsize=(14, 10))
    corr_alz = df_alz[numerical_alz + ["Diagnosis"]].corr()
    mask_up = np.triu(np.ones_like(corr_alz, dtype=bool))
    sns.heatmap(corr_alz, mask=mask_up, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, ax=ax, linewidths=0.4,
                annot_kws={"size": 7})
    ax.set_title("Correlação — Alzheimer Dataset")
    save_and_show(fig, "alz_correlacao")


# ============================================================================
# PÁGINA: PRÉ-PROCESSAMENTO
# ============================================================================

elif pagina == "⚙️  Pré-processamento":

    st.markdown('<p class="section-title">⚙️ Pré-processamento — Pipeline Completo</p>',
                unsafe_allow_html=True)
    divider()

    caixa_conceito("""
    <b>Estratégia de Pré-processamento:</b><br>
    1. Separar features numéricas (contínuas) e categóricas/binárias<br>
    2. Aplicar <code>StandardScaler</code> nas numéricas<br>
    3. Aplicar <code>OneHotEncoder</code> nas categóricas (variáveis com mais de 2 valores)<br>
    4. Features binárias (0/1): passthrough ou incluir no OneHotEncoder<br>
    5. Tudo encapsulado em <code>ColumnTransformer</code> dentro de um <code>Pipeline</code>
    """)

    with st.spinner("Carregando dataset..."):
        df_alz = carregar_alzheimer()

    X_alz = df_alz.drop("Diagnosis", axis=1)
    y_alz = df_alz["Diagnosis"]

    numerical_alz = [col for col in X_alz.columns if X_alz[col].nunique() > 10]
    categorical_alz = [col for col in X_alz.columns if X_alz[col].nunique() <= 10]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_alz, y_alz, test_size=0.2, random_state=42, stratify=y_alz
    )

    secao("Split Treino / Teste")
    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Total", len(X_alz))
    col_s2.metric("Treino (80%)", len(X_tr))
    col_s3.metric("Teste (20%)", len(X_te))

    caixa_dica("""
    Usamos <code>stratify=y</code> para garantir que a proporção das classes
    seja mantida tanto no treino quanto no teste — especialmente importante
    quando as classes são desbalanceadas.
    """)

    divider()
    secao("ColumnTransformer — Pré-processamento Diferenciado")

    caixa_conceito(f"""
    <b>Features numéricas ({len(numerical_alz)}):</b> StandardScaler<br>
    <b>Features categóricas ({len(categorical_alz)}):</b> OneHotEncoder<br><br>
    O <code>ColumnTransformer</code> aplica transformações diferentes para
    cada grupo de features e concatena o resultado em uma única matriz.
    """)

    with st.expander("Ver os grupos de features"):
        col_fn, col_fc = st.columns(2)
        with col_fn:
            st.markdown("**Numéricas:**")
            for f in numerical_alz:
                st.markdown(f"- `{f}`")
        with col_fc:
            st.markdown("**Categóricas/Binárias:**")
            for f in categorical_alz:
                nuniq = X_alz[f].nunique()
                st.markdown(f"- `{f}` ({nuniq} valores únicos)")

    divider()
    secao("Exemplo de Pipeline com Random Forest")

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numerical_alz),
        ("cat", OneHotEncoder(drop="first", sparse_output=False,
                               handle_unknown="ignore"), categorical_alz),
    ])

    pipe_demo = Pipeline([("pre", preprocessor),
                          ("clf", RandomForestClassifier(n_estimators=50, random_state=42))])
    pipe_demo.fit(X_tr, y_tr)
    y_pred_demo = pipe_demo.predict(X_te)
    acc_demo = accuracy_score(y_te, y_pred_demo)
    f1_demo  = f1_score(y_te, y_pred_demo, average="macro")

    col_m1, col_m2 = st.columns(2)
    col_m1.metric("Accuracy (teste)", f"{acc_demo:.4f}")
    col_m2.metric("F1-macro (teste)", f"{f1_demo:.4f}")

    with st.expander("Ver código Python — Pré-processamento"):
        code_block("""
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Separar features por tipo
numerical   = [col for col in X.columns if X[col].nunique() > 10]
categorical = [col for col in X.columns if X[col].nunique() <= 10]

# ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical),
    ('cat', OneHotEncoder(drop='first', sparse_output=False,
                           handle_unknown='ignore'), categorical),
])

# Pipeline completo
pipe = Pipeline([
    ('pre', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
pipe.fit(X_train, y_train)
""")


# ============================================================================
# PÁGINA: TREINAMENTO DOS MODELOS
# ============================================================================

elif pagina == "🤖  Treinamento dos Modelos":

    st.markdown('<p class="section-title">🤖 Treinamento — Todos os Modelos</p>',
                unsafe_allow_html=True)
    divider()

    caixa_conceito("""
    <b>Treinamos 6 modelos com o mesmo Pipeline e avaliamos com cross-validation (cv=10).</b><br>
    Cada modelo recebe o mesmo pré-processamento. As métricas são calculadas
    nas 10 dobras para garantir robustez na comparação.
    """)

    with st.spinner("Carregando dataset Alzheimer..."):
        try:
            df_alz = carregar_alzheimer()
        except Exception as e:
            st.error(f"Erro: {e}")
            st.stop()

    X_alz = df_alz.drop("Diagnosis", axis=1)
    y_alz = df_alz["Diagnosis"]

    numerical_alz = [col for col in X_alz.columns if X_alz[col].nunique() > 10]
    categorical_alz = [col for col in X_alz.columns if X_alz[col].nunique() <= 10]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numerical_alz),
        ("cat", OneHotEncoder(drop="first", sparse_output=False,
                               handle_unknown="ignore"), categorical_alz),
    ])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_alz, y_alz, test_size=0.2, random_state=42, stratify=y_alz
    )

    modelos_config = [
        ("KNN",                KNeighborsClassifier()),
        ("SVM",                SVC(probability=True)),
        ("Decision Tree",      DecisionTreeClassifier(max_depth=10, random_state=42)),
        ("Random Forest",      RandomForestClassifier(n_estimators=100, random_state=42)),
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
    ]
    if XGBOOST_OK:
        modelos_config.append(("XGBoost", XGBClassifier(random_state=42, eval_metric="logloss")))

    modelo_sel = st.selectbox(
        "Selecione o modelo para ver detalhes:",
        [m[0] for m in modelos_config]
    )

    clf_sel = dict(modelos_config)[modelo_sel]
    pipe_sel = Pipeline([("pre", preprocessor), ("clf", clf_sel)])

    with st.spinner(f"Treinando {modelo_sel}..."):
        pipe_sel.fit(X_tr, y_tr)
        y_pred_sel = pipe_sel.predict(X_te)
        acc_sel = accuracy_score(y_te, y_pred_sel)
        f1_sel  = f1_score(y_te, y_pred_sel, average="macro")

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Accuracy (teste)", f"{acc_sel:.4f}")
    col_m2.metric("F1-macro (teste)", f"{f1_sel:.4f}")
    col_m3.metric("Modelo", modelo_sel)

    divider()
    secao("Relatório de Classificação")
    labels_diag = ["Não diagnosticado", "Diagnosticado"]
    report_sel = classification_report(y_te, y_pred_sel,
                                       target_names=labels_diag, output_dict=True)
    st.dataframe(pd.DataFrame(report_sel).T.round(4), use_container_width=True)

    secao("Matriz de Confusão")
    cm_sel = confusion_matrix(y_te, y_pred_sel)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_sel, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels_diag, yticklabels=labels_diag, linewidths=1)
    ax.set_xlabel("Previsto"); ax.set_ylabel("Real")
    ax.set_title(f"Matriz de Confusão — {modelo_sel}")
    save_and_show(fig, f"alz_cm_{modelo_sel.replace(' ', '_')}")

    divider()
    secao("Cross-Validation (cv=10)")
    with st.spinner(f"Executando cross-validation para {modelo_sel}..."):
        cv_sel = cross_validate(pipe_sel, X_alz, y_alz, cv=10,
                                scoring=["accuracy", "f1_macro"])

    col_cv1, col_cv2 = st.columns(2)
    col_cv1.metric("CV Accuracy",
                   f"{cv_sel['test_accuracy'].mean():.4f} ± {cv_sel['test_accuracy'].std():.4f}")
    col_cv2.metric("CV F1-macro",
                   f"{cv_sel['test_f1_macro'].mean():.4f} ± {cv_sel['test_f1_macro'].std():.4f}")


# ============================================================================
# PÁGINA: COMPARATIVO FINAL
# ============================================================================

elif pagina == "🏆  Comparativo Final":

    st.markdown('<p class="section-title">🏆 Comparativo Final — Todos os Modelos</p>',
                unsafe_allow_html=True)
    divider()

    caixa_dica("""
    Este painel treina e avalia <b>todos os modelos em sequência</b> com cross-validation cv=10.
    Pode demorar 1-2 minutos. Os resultados mostram qual algoritmo performa melhor
    no diagnóstico de Alzheimer.
    """)

    with st.spinner("Carregando dataset..."):
        try:
            df_alz = carregar_alzheimer()
        except Exception as e:
            st.error(f"Erro: {e}")
            st.stop()

    X_alz = df_alz.drop("Diagnosis", axis=1)
    y_alz = df_alz["Diagnosis"]
    numerical_alz = [col for col in X_alz.columns if X_alz[col].nunique() > 10]
    categorical_alz = [col for col in X_alz.columns if X_alz[col].nunique() <= 10]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numerical_alz),
        ("cat", OneHotEncoder(drop="first", sparse_output=False,
                               handle_unknown="ignore"), categorical_alz),
    ])

    modelos_comp = [
        ("KNN",                KNeighborsClassifier()),
        ("SVM",                SVC()),
        ("Decision Tree",      DecisionTreeClassifier(max_depth=10, random_state=42)),
        ("Random Forest",      RandomForestClassifier(n_estimators=100, random_state=42)),
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
    ]
    if XGBOOST_OK:
        modelos_comp.append(("XGBoost", XGBClassifier(random_state=42, eval_metric="logloss")))

    if st.button("▶️ Executar comparativo completo (cv=10)", type="primary"):

        resultados = []
        progress = st.progress(0, text="Iniciando...")

        for idx, (nome, clf) in enumerate(modelos_comp):
            progress.progress((idx) / len(modelos_comp), text=f"Treinando {nome}...")
            pipe_c = Pipeline([("pre", preprocessor), ("clf", clf)])
            cv_c = cross_validate(pipe_c, X_alz, y_alz, cv=10,
                                  scoring=["accuracy", "f1_macro"])
            resultados.append({
                "Modelo": nome,
                "Accuracy": cv_c["test_accuracy"].mean(),
                "Accuracy ±": cv_c["test_accuracy"].std(),
                "F1-macro": cv_c["test_f1_macro"].mean(),
                "F1-macro ±": cv_c["test_f1_macro"].std(),
            })

        progress.progress(1.0, text="Concluído!")

        df_res = pd.DataFrame(resultados).sort_values("Accuracy", ascending=False)
        st.dataframe(df_res.round(4), use_container_width=True)

        # Gráfico comparativo
        fig, ax = plt.subplots(figsize=(11, 5))
        nomes = df_res["Modelo"].tolist()
        accs  = df_res["Accuracy"].tolist()
        f1s   = df_res["F1-macro"].tolist()
        stds  = df_res["Accuracy ±"].tolist()
        x_pos = np.arange(len(nomes))

        ax.bar(x_pos - 0.2, accs, width=0.35, label="Accuracy", color="#667eea", alpha=0.9)
        ax.bar(x_pos + 0.2, f1s,  width=0.35, label="F1-macro",  color="#38a169", alpha=0.9)
        ax.errorbar(x_pos - 0.2, accs, yerr=stds, fmt="none",
                    color="black", capsize=4, linewidth=1.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(nomes, rotation=15, ha="right", fontsize=10)
        ax.set_ylim(max(0, min(accs) - 0.05), min(1.0, max(accs) + 0.05))
        ax.set_ylabel("Métrica (cv=10)")
        ax.set_title("Comparativo de Modelos — Alzheimer (CV=10)")
        ax.legend(); ax.grid(alpha=0.3, axis="y")
        for i, (acc, f1) in enumerate(zip(accs, f1s)):
            ax.text(i - 0.2, acc + 0.003, f"{acc:.3f}", ha="center", fontsize=8)
            ax.text(i + 0.2, f1  + 0.003, f"{f1:.3f}",  ha="center", fontsize=8)
        save_and_show(fig, "alz_comparativo_completo")

        best = df_res.iloc[0]
        caixa_sucesso(f"""
        <b>🏆 Melhor modelo: {best['Modelo']}</b><br>
        Accuracy: <b>{best['Accuracy']:.4f} ± {best['Accuracy ±']:.4f}</b><br>
        F1-macro: <b>{best['F1-macro']:.4f} ± {best['F1-macro ±']:.4f}</b>
        """)

    else:
        st.info("Clique no botão acima para executar o comparativo completo.")


# ============================================================================
# PÁGINA: ENSEMBLE
# ============================================================================

elif pagina == "🎯  Ensemble — VotingClassifier":

    st.markdown('<p class="section-title">🎯 Ensemble — VotingClassifier</p>',
                unsafe_allow_html=True)
    divider()

    caixa_conceito("""
    <b>O que é um Ensemble Voting?</b><br>
    Combina as previsões de <b>múltiplos modelos independentes</b>.
    Em <code>voting='soft'</code>, cada modelo emite uma probabilidade e
    a classe com maior probabilidade média vence. Geralmente supera qualquer
    modelo individual — desde que os modelos errem em momentos diferentes.
    """)

    divider()
    secao("Votação Suave (Soft Voting) vs Dura (Hard Voting)")

    col_hard, col_soft = st.columns(2)
    with col_hard:
        caixa_conceito("""
        <b>Hard Voting</b><br>
        Cada modelo emite uma <b>classe</b>.<br>
        A classe que aparece na maioria dos modelos vence.<br>
        Simples, mas perde a informação de confiança.<br>
        <code>voting='hard'</code>
        """)
    with col_soft:
        caixa_conceito("""
        <b>Soft Voting</b><br>
        Cada modelo emite uma <b>probabilidade</b>.<br>
        A classe com a maior probabilidade <b>média</b> vence.<br>
        Melhor performance — aproveita a confiança do modelo.<br>
        <code>voting='soft'</code> — requer <code>predict_proba</code>
        """)

    with st.spinner("Carregando dataset..."):
        try:
            df_alz = carregar_alzheimer()
        except Exception as e:
            st.error(f"Erro: {e}")
            st.stop()

    X_alz = df_alz.drop("Diagnosis", axis=1)
    y_alz = df_alz["Diagnosis"]
    numerical_alz = [col for col in X_alz.columns if X_alz[col].nunique() > 10]
    categorical_alz = [col for col in X_alz.columns if X_alz[col].nunique() <= 10]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numerical_alz),
        ("cat", OneHotEncoder(drop="first", sparse_output=False,
                               handle_unknown="ignore"), categorical_alz),
    ])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_alz, y_alz, test_size=0.2, random_state=42, stratify=y_alz
    )

    divider()
    secao("Construindo o Ensemble — Top 3 Modelos")

    caixa_dica("""
    Selecionamos os 3 modelos com melhor trade-off accuracy × diversidade:
    <b>SVM + Random Forest + Logistic Regression</b>. A diversidade entre os modelos
    é crucial — modelos muito similares não trazem ganho real ao ensemble.
    """)

    clf1 = Pipeline([("pre", preprocessor), ("clf", SVC(probability=True))])
    clf2 = Pipeline([("pre", preprocessor), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])
    clf3 = Pipeline([("pre", preprocessor), ("clf", LogisticRegression(max_iter=1000, random_state=42))])

    if XGBOOST_OK:
        clf4 = Pipeline([("pre", preprocessor), ("clf", XGBClassifier(random_state=42, eval_metric="logloss"))])
        estimators_ens = [("svm", clf1), ("rf", clf2), ("lr", clf3), ("xgb", clf4)]
    else:
        estimators_ens = [("svm", clf1), ("rf", clf2), ("lr", clf3)]

    ensemble = VotingClassifier(estimators=estimators_ens, voting="soft")

    with st.spinner("Treinando ensemble (pode demorar)..."):
        ensemble.fit(X_tr, y_tr)
        y_pred_ens = ensemble.predict(X_te)
        acc_ens = accuracy_score(y_te, y_pred_ens)
        f1_ens  = f1_score(y_te, y_pred_ens, average="macro")

    col_e1, col_e2 = st.columns(2)
    col_e1.metric("Accuracy — Ensemble", f"{acc_ens:.4f}")
    col_e2.metric("F1-macro — Ensemble", f"{f1_ens:.4f}")

    divider()
    secao("Comparando com Modelos Individuais")

    with st.spinner("Comparando com modelos individuais (cv=10)..."):
        resultados_comp = []
        for nome, pipe_c in estimators_ens:
            cv_c = cross_validate(pipe_c, X_alz, y_alz, cv=10,
                                  scoring=["accuracy", "f1_macro"])
            resultados_comp.append({
                "Modelo": nome.upper(),
                "Accuracy (CV)": cv_c["test_accuracy"].mean(),
                "F1-macro (CV)": cv_c["test_f1_macro"].mean(),
            })
        # Ensemble
        cv_ens = cross_validate(ensemble, X_alz, y_alz, cv=10,
                                scoring=["accuracy", "f1_macro"])
        resultados_comp.append({
            "Modelo": "🏆 ENSEMBLE",
            "Accuracy (CV)": cv_ens["test_accuracy"].mean(),
            "F1-macro (CV)": cv_ens["test_f1_macro"].mean(),
        })

    df_comp_ens = pd.DataFrame(resultados_comp).sort_values("Accuracy (CV)", ascending=False)
    st.dataframe(df_comp_ens.round(4), use_container_width=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    cores_ens = ["#667eea"] * (len(df_comp_ens) - 1) + ["#f59e0b"]
    bars = ax.bar(df_comp_ens["Modelo"], df_comp_ens["Accuracy (CV)"],
                  color=cores_ens, alpha=0.9)
    ax.set_ylabel("Accuracy (CV=10)")
    ax.set_title("Ensemble vs Modelos Individuais — Alzheimer")
    ax.set_ylim(max(0, df_comp_ens["Accuracy (CV)"].min() - 0.02), 1.0)
    for bar, val in zip(bars, df_comp_ens["Accuracy (CV)"]):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.002,
                f"{val:.3f}", ha="center", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    save_and_show(fig, "alz_ensemble_comparativo")

    divider()
    with st.expander("Ver código Python — Ensemble VotingClassifier"):
        code_block("""
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

clf1 = Pipeline([('pre', preprocessor), ('clf', SVC(probability=True))])
clf2 = Pipeline([('pre', preprocessor), ('clf', RandomForestClassifier(n_estimators=100, random_state=42))])
clf3 = Pipeline([('pre', preprocessor), ('clf', LogisticRegression(max_iter=1000, random_state=42))])

# Soft Voting — usa probabilidades
ensemble = VotingClassifier(
    estimators=[('svm', clf1), ('rf', clf2), ('lr', clf3)],
    voting='soft'
)

ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)
print(classification_report(y_test, y_pred))

# Cross-validation
cv = cross_validate(ensemble, X, y, cv=10, scoring=['accuracy', 'f1_macro'])
print(f"Accuracy: {cv['test_accuracy'].mean():.4f} ± {cv['test_accuracy'].std():.4f}")
""")


# ============================================================================
# PÁGINA: SALVAR MODELO
# ============================================================================

elif pagina == "💾  Salvar e Carregar Modelo":

    st.markdown('<p class="section-title">💾 Salvar e Carregar Modelo com joblib</p>',
                unsafe_allow_html=True)
    divider()

    caixa_conceito("""
    <b>Por que salvar o modelo?</b><br>
    Treinar um modelo pode levar minutos ou horas. Em produção, você salva o modelo
    treinado uma vez e o carrega quando precisar fazer previsões — sem retreinar.
    O <code>joblib</code> é a forma recomendada para salvar modelos scikit-learn.
    """)

    divider()
    secao("Como funciona o joblib")

    col_save, col_load = st.columns(2)
    with col_save:
        caixa_conceito("""
        <b>Salvar o modelo:</b><br>
        <code>import joblib</code><br>
        <code>joblib.dump(model, 'model.joblib')</code><br><br>
        Salva o Pipeline completo — pré-processador + classificador.
        Ao carregar, o modelo já sabe transformar novos dados automaticamente.
        """)
    with col_load:
        caixa_conceito("""
        <b>Carregar o modelo:</b><br>
        <code>model = joblib.load('model.joblib')</code><br><br>
        O modelo carregado é idêntico ao original.
        Pode fazer predições em novos dados sem retreinar.
        """)

    divider()
    secao("Demonstração — Treinar, Salvar e Carregar")

    with st.spinner("Treinando Random Forest no dataset Alzheimer..."):
        try:
            df_alz = carregar_alzheimer()
        except Exception as e:
            st.error(f"Erro: {e}")
            st.stop()

    X_alz = df_alz.drop("Diagnosis", axis=1)
    y_alz = df_alz["Diagnosis"]
    numerical_alz = [col for col in X_alz.columns if X_alz[col].nunique() > 10]
    categorical_alz = [col for col in X_alz.columns if X_alz[col].nunique() <= 10]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numerical_alz),
        ("cat", OneHotEncoder(drop="first", sparse_output=False,
                               handle_unknown="ignore"), categorical_alz),
    ])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_alz, y_alz, test_size=0.2, random_state=42, stratify=y_alz
    )

    pipe_final = Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipe_final.fit(X_tr, y_tr)
    acc_final = accuracy_score(y_te, pipe_final.predict(X_te))

    import joblib
    model_path = os.path.join(OUTPUTS_DIR, "random_forest_alzheimer.joblib")
    joblib.dump(pipe_final, model_path)

    caixa_sucesso(f"""
    <b>Modelo treinado e salvo com sucesso!</b><br>
    Accuracy no teste: <b>{acc_final:.4f}</b><br>
    Arquivo salvo em: <code>{model_path}</code>
    """)

    # Carregar o modelo salvo
    model_carregado = joblib.load(model_path)
    y_pred_carregado = model_carregado.predict(X_te)
    acc_carregado = accuracy_score(y_te, y_pred_carregado)

    caixa_sucesso(f"""
    <b>Modelo carregado com sucesso!</b><br>
    Accuracy após carregar: <b>{acc_carregado:.4f}</b><br>
    (Deve ser igual ao anterior — {acc_final:.4f})
    """)

    divider()
    with st.expander("Ver código Python — Salvar e Carregar Modelo"):
        code_block("""
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Treinar o modelo
pipe = Pipeline([
    ('pre', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])
pipe.fit(X_train, y_train)

# Salvar
joblib.dump(pipe, 'random_forest_alzheimer.joblib')
print("Modelo salvo!")

# Carregar (em outro script, ou em produção)
modelo_prod = joblib.load('random_forest_alzheimer.joblib')

# Fazer predições sem retreinar
novos_pacientes = X_test.iloc[:5]
predicoes = modelo_prod.predict(novos_pacientes)
probabilidades = modelo_prod.predict_proba(novos_pacientes)

for i, (pred, prob) in enumerate(zip(predicoes, probabilidades)):
    status = "Diagnosticado" if pred == 1 else "Não diagnosticado"
    print(f"Paciente {i+1}: {status} (confiança: {max(prob)*100:.1f}%)")
""")

    divider()
    st.markdown("""
    <div class="footer">
        <p style="margin:0; font-size:1.1rem; font-weight:700;">Machine Learning — Aula 08</p>
        <p style="margin:0.3rem 0 0; font-size:0.9rem; opacity:0.8;">
            Comparação de Modelos &nbsp;·&nbsp;
            Cláudio Ferreira Neves &nbsp;·&nbsp; SENAI/SC
        </p>
    </div>
    """, unsafe_allow_html=True)
