"""
=============================================================================
Machine Learning — Aula 06 / Decision Tree, Random Forest e XGBoost
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

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score,
)
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif

try:
    from xgboost import XGBClassifier
    XGBOOST_OK = True
except ImportError:
    XGBOOST_OK = False

PAGE_PORTAL  = "pages/Portal.py"
PAGE_AULA_05 = "pages/Aula_05.py"
PAGE_AULA_07 = "pages/Aula_07.py"

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────
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

# ── dados ─────────────────────────────────────────────────────────────────────
@st.cache_data
def carregar_breast_cancer():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target_names[data.target]
    return df, data.target_names.tolist()

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Decision Tree e Random Forest",
                   page_icon="🌳", layout="wide", initial_sidebar_state="expanded")

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
    st.markdown('<div class="sidebar-header"><h2 style="margin:0;font-size:1.3rem;">🌳 Aula 06</h2>'
                '<p style="margin:.4rem 0 0;font-size:.8rem;opacity:.8;">Decision Tree e Ensemble</p></div>',
                unsafe_allow_html=True)
    pagina = st.radio("Nav", [
        "🏠  Início",
        "🌳  Decision Tree — Teoria",
        "🏥  Decision Tree — Breast Cancer",
        "🌲  Random Forest",
        "⚡  XGBoost",
        "📊  Comparativo de Modelos",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("<div style='font-size:.75rem;color:#888;text-align:center;'>"
                "📁 Gráficos em <code>outputs/</code></div>", unsafe_allow_html=True)

# ── navegação entre aulas ────────────────────────────────────────────────────
from nav import tab_nav
tab_nav(6)  # replace CURRENT_NUMBER with the correct int

st.markdown("<p style='text-align:center;font-size:.95rem;color:#667eea;font-weight:600;margin-bottom:0;'>"
            "Autor: Especialista Cláudio Ferreira Neves</p>", unsafe_allow_html=True)
nl, nm, nr = st.columns([1.2, 4, 1.2])
with nl:
    if st.button("← Aula 05", use_container_width=True, key="nav_prev"):
        st.switch_page(PAGE_AULA_05)
with nm:
    if st.button("🏠 Portal", use_container_width=True, key="nav_portal"):
        st.switch_page(PAGE_PORTAL)
with nr:
    if st.button("Aula 07 →", use_container_width=True, key="nav_next"):
        st.switch_page(PAGE_AULA_07)

# ============================================================================
# INÍCIO
# ============================================================================
if pagina == "🏠  Início":
    st.markdown("<div style='text-align:center;padding:2rem 0 1rem;'>"
                "<h1 style='font-size:2.8rem;font-weight:800;color:#1a1a2e;'>🌳 Decision Tree, Random Forest e XGBoost</h1>"
                "<p style='font-size:1.15rem;color:#555;max-width:740px;margin:0 auto;'>"
                "Da árvore simples ao ensemble poderoso: aprenda como combinar múltiplos modelos "
                "para obter resultados superiores — e por que o XGBoost domina o mundo real.</p></div>",
                unsafe_allow_html=True)
    divider()
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown('<div class="metric-card"><h2>3</h2><p>Algoritmos</p></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="metric-card"><h2>🏥</h2><p>Breast Cancer</p></div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="metric-card"><h2>🌲</h2><p>Ensemble</p></div>', unsafe_allow_html=True)
    with c4: st.markdown('<div class="metric-card"><h2>📊</h2><p>Feature Importance</p></div>', unsafe_allow_html=True)
    divider()
    st.markdown("## 📚 O que você vai aprender")
    ca, cb = st.columns(2)
    with ca:
        caixa("<b>🌳 Decision Tree (Árvore de Decisão)</b><br>"
              "Aprende regras de decisão a partir dos dados. Cada nó faz uma pergunta "
              "baseada em uma feature (ex: <i>glicose > 130?</i>). Altamente interpretável — "
              "você pode visualizar cada regra aprendida. Cuidado com <b>overfitting</b> "
              "quando a árvore cresce demais.")
        caixa("<b>📊 Feature Importance</b><br>"
              "Árvores e florestas calculam automaticamente a <b>importância de cada feature</b> "
              "— qual variável mais reduziu a impureza (Gini/Entropia) no treinamento. "
              "Ferramenta poderosa para seleção de variáveis e interpretabilidade.",
              cor="#fff8e1", borda="#f59e0b")
        caixa("<b>🌲 Random Forest (Floresta Aleatória)</b><br>"
              "Ensemble de <b>múltiplas árvores</b> treinadas com amostras aleatórias "
              "(<b>bagging</b>) e subconjuntos aleatórios de features. Reduz a variância "
              "do modelo individual — mais robusto e preciso.")
    with cb:
        caixa("<b>⚡ XGBoost (Extreme Gradient Boosting)</b><br>"
              "Ensemble de árvores treinadas <b>sequencialmente</b>. Cada nova árvore "
              "corrige os erros da anterior (<b>boosting</b>). Estado da arte em tabular "
              "data — vencedor de dezenas de competições Kaggle.")
        caixa("<b>🔄 Bagging vs Boosting</b><br>"
              "• <b>Bagging</b> (RF): árvores treinadas em <b>paralelo</b> com amostras "
              "aleatórias. Reduz variância.<br>"
              "• <b>Boosting</b> (XGB): árvoras treinadas em <b>sequência</b>. "
              "Cada uma foca nos erros anteriores. Reduz viés.")
        caixa("<b>🏥 Dataset Breast Cancer</b><br>"
              "569 tumores mamários (benignos/malignos) com 30 features numéricas. "
              "Dataset clássico para classificação binária — usado para demonstrar "
              "Decision Tree, Random Forest e XGBoost com comparativo completo.")
    divider()
    st.markdown("""<div class="footer">
        <p style="margin:0;font-size:1.1rem;font-weight:700;">Machine Learning — Aula 06</p>
        <p style="margin:.3rem 0 0;font-size:.9rem;opacity:.8;">
        Decision Tree, Random Forest e XGBoost · Cláudio Ferreira Neves · SENAI/SC</p></div>""",
        unsafe_allow_html=True)

# ============================================================================
# DECISION TREE — TEORIA
# ============================================================================
elif pagina == "🌳  Decision Tree — Teoria":
    st.markdown('<h2>🌳 Decision Tree — Fundamentos</h2>', unsafe_allow_html=True)
    divider()

    caixa("<b>Como funciona uma Árvore de Decisão?</b><br>"
          "Constrói um modelo recursivamente, dividindo os dados em subconjuntos "
          "<b>cada vez mais homogêneos</b>. Em cada nó, escolhe a feature e o "
          "limiar que melhor separaram as classes. O critério padrão é o "
          "<b>índice de Gini</b>.")

    divider()
    secao("Critérios de Divisão")
    ct1, ct2 = st.columns(2)
    with ct1:
        st.markdown(r"""
**Índice de Gini:**
$$\text{Gini}(t) = 1 - \sum_{k=1}^{K} p_k^2$$

- $p_k$ = proporção de amostras da classe $k$ no nó $t$
- **Gini = 0**: nó puro (só uma classe) ✅
- **Gini ≈ 0.5**: máxima impureza (50%/50%)

**Entropia (critério alternativo):**
$$H(t) = -\sum_{k=1}^{K} p_k \log_2(p_k)$$

- Entropia = 0: nó puro ✅
- Entropia = 1: máxima impureza
""")
    with ct2:
        caixa("<b>O que o Nó de uma Árvore mostra?</b><br><br>"
              "• <b>Pergunta</b>: ex. <code>worst concave points ≤ 0.131</code><br>"
              "• <b>Impureza (gini)</b>: 0 = puro, 0.5 = mistura máxima<br>"
              "• <b>Samples</b>: quantas amostras chegaram nesse nó<br>"
              "• <b>Value</b>: distribuição por classe<br>"
              "• <b>Class</b>: classe prevista (a mais frequente)<br><br>"
              "Cores: 🟠 benign &nbsp;|&nbsp; 🔵 malignant")
        caixa("<b>⚠️ Árvores NÃO precisam de escalonamento</b><br>"
              "Diferente do KNN e SVM, a Decision Tree usa os valores apenas como "
              "<b>pontos de corte</b>, não como distâncias. "
              "O <code>StandardScaler</code> é desnecessário aqui.",
              cor="#fff0f0", borda="#e53e3e")

    divider()
    secao("Controle do Overfitting — Simulação Interativa")

    caixa("<b>Overfitting em Árvores</b><br>"
          "Sem restrições, a árvore cresce até memorizar <b>cada ponto de treino</b>. "
          "O parâmetro <code>max_depth</code> limita a profundidade, controlando "
          "o tradeoff viés-variância.", cor="#fff8e1", borda="#f59e0b")

    max_d = st.slider("max_depth:", 1, 20, 5, 1, key="md_dt")

    with st.spinner("Treinando árvore..."):
        try:
            df_bc, _ = carregar_breast_cancer()
            X_bc = df_bc.drop("target", axis=1)
            y_bc = df_bc["target"]
            Xtr_bc, Xte_bc, ytr_bc, yte_bc = train_test_split(
                X_bc, y_bc, test_size=0.2, stratify=y_bc, random_state=42)
            dt_sim = DecisionTreeClassifier(max_depth=max_d, random_state=42)
            dt_sim.fit(Xtr_bc, ytr_bc)
            acc_tr = accuracy_score(ytr_bc, dt_sim.predict(Xtr_bc))
            acc_te = accuracy_score(yte_bc, dt_sim.predict(Xte_bc))
            n_nos  = dt_sim.tree_.node_count
        except:
            acc_tr, acc_te, n_nos = 0.0, 0.0, 0

    cs1, cs2, cs3 = st.columns(3)
    cs1.metric("Acc — Treino",  f"{acc_tr:.4f}")
    cs2.metric("Acc — Teste",   f"{acc_te:.4f}", delta=f"{acc_te-acc_tr:.4f}")
    cs3.metric("Nós na árvore", str(n_nos))

    if acc_tr - acc_te > 0.05:
        caixa(f"<b>⚠️ Possível overfitting detectado!</b><br>"
              f"A acurácia de treino ({acc_tr:.3f}) é bem maior que a de teste "
              f"({acc_te:.3f}). Tente reduzir o max_depth.",
              cor="#fff0f0", borda="#e53e3e")
    else:
        caixa(f"<b>✅ Bom equilíbrio!</b><br>Diferença treino/teste: "
              f"{acc_tr-acc_te:.4f}. max_depth={max_d} parece razoável.",
              cor="#f0fff4", borda="#38a169")

    divider()
    secao("Curva de Overfitting: max_depth vs Accuracy")
    with st.spinner("Calculando curvas (1 a 20)..."):
        try:
            depths = range(1, 21)
            acc_train_list, acc_test_list = [], []
            for d in depths:
                dt_cv = DecisionTreeClassifier(max_depth=d, random_state=42)
                dt_cv.fit(Xtr_bc, ytr_bc)
                acc_train_list.append(accuracy_score(ytr_bc, dt_cv.predict(Xtr_bc)))
                acc_test_list.append(accuracy_score(yte_bc, dt_cv.predict(Xte_bc)))
            curva_ok = True
        except:
            curva_ok = False

    if curva_ok:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(depths, acc_train_list, "b-o", markersize=5, linewidth=2, label="Treino")
        ax.plot(depths, acc_test_list,  "r-s", markersize=5, linewidth=2, label="Teste")
        ax.axvline(max_d, color="gold", linestyle="--", linewidth=2,
                   label=f"max_depth={max_d} selecionado")
        ax.set_xlabel("max_depth"); ax.set_ylabel("Accuracy")
        ax.set_title("Curva Treino vs Teste — Decision Tree (Breast Cancer)")
        ax.legend(); ax.grid(alpha=0.3)
        save_and_show(fig, "dt_overfitting_curva")

    divider()
    with st.expander("Ver código Python — Decision Tree"):
        code_block("""
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# Árvores NÃO precisam de StandardScaler
pipe_dt = Pipeline(steps=[
    ('dt', DecisionTreeClassifier(criterion='gini', max_depth=10, random_state=42))
])

pipe_dt.fit(X_train, y_train)
y_pred = pipe_dt.predict(X_test)

# Visualizar a árvore (profundidade máxima 3 para legibilidade)
dt = pipe_dt.named_steps['dt']
plt.figure(figsize=(16, 8))
plot_tree(dt, filled=True, feature_names=X.columns,
          class_names=dt.classes_, max_depth=3)
plt.show()

# Feature Importance
importances = pd.Series(dt.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).head(10).plot(kind='barh')
plt.title("Top 10 Features — Decision Tree")
plt.show()
""")

# ============================================================================
# DECISION TREE — BREAST CANCER
# ============================================================================
elif pagina == "🏥  Decision Tree — Breast Cancer":
    st.markdown('<h2>🏥 Decision Tree — Dataset Breast Cancer</h2>', unsafe_allow_html=True)
    divider()

    caixa("<b>Breast Cancer Wisconsin Dataset</b><br>"
          "569 tumores mamários: 357 benignos, 212 malignos. "
          "30 features numéricas derivadas de imagens digitalizadas de tecido mamário "
          "(raio, textura, perímetro, área, etc.). "
          "Target binário: <code>benign</code> vs <code>malignant</code>.")

    with st.spinner("Carregando dataset..."):
        df_bc, target_names = carregar_breast_cancer()
    st.success(f"Dataset: {df_bc.shape[0]} amostras × {df_bc.shape[1]} colunas")

    secao("Análise Exploratória")
    cbc1, cbc2 = st.columns(2)
    with cbc1:
        st.markdown("**Distribuição do target:**")
        vc = df_bc["target"].value_counts().reset_index()
        vc.columns = ["Classe","Contagem"]
        st.dataframe(vc, use_container_width=True)
        pct_malig = (df_bc["target"]=="malignant").mean()*100
        st.info(f"Malignant: {pct_malig:.1f}%")
    with cbc2:
        st.markdown("**Primeiras linhas:**")
        st.dataframe(df_bc.head(6), use_container_width=True)

    divider()
    secao("Heatmap de Correlação")
    fig, ax = plt.subplots(figsize=(14, 12))
    corr = df_bc.drop("target", axis=1).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=.3, annot_kws={"size": 6})
    ax.set_title("Correlação — Breast Cancer")
    save_and_show(fig, "bc_correlacao")

    divider()
    X_bc = df_bc.drop("target", axis=1); y_bc = df_bc["target"]
    numerical_bc = X_bc.columns.tolist()

    max_d = st.slider("max_depth da árvore:", 1, 20, 10, 1, key="bc_depth")

    Xtr, Xte, ytr, yte = train_test_split(
        X_bc, y_bc, test_size=0.2, stratify=y_bc, random_state=42)

    pipe_dt = Pipeline([("dt", DecisionTreeClassifier(
        criterion="gini", max_depth=max_d, random_state=42))])
    pipe_dt.fit(Xtr, ytr); ypred = pipe_dt.predict(Xte)
    acc_d = accuracy_score(yte, ypred); f1_d = f1_score(yte, ypred, average="macro")

    secao("Resultados — Decision Tree")
    cd1, cd2, cd3 = st.columns(3)
    cd1.metric("Accuracy", f"{acc_d:.4f}"); cd2.metric("F1-macro", f"{f1_d:.4f}")
    cd3.metric("Amostras teste", len(yte))

    st.dataframe(pd.DataFrame(classification_report(yte,ypred,output_dict=True)).T.round(4),
                 use_container_width=True)

    secao("Visualização da Árvore (max_depth=3 para legibilidade)")
    dt_model = pipe_dt.named_steps["dt"]
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(dt_model, filled=True, feature_names=X_bc.columns,
              class_names=dt_model.classes_, max_depth=3, ax=ax,
              fontsize=9)
    ax.set_title("Decision Tree — Breast Cancer (max_depth=3)", fontsize=13)
    save_and_show(fig, "bc_dt_arvore")

    secao("Feature Importance — Top 15")
    imp = pd.DataFrame({
        "feature": X_bc.columns,
        "importance": dt_model.feature_importances_
    }).sort_values("importance", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp["feature"][:15], imp["importance"][:15], color="#667eea")
    ax.set_xlabel("Importância"); ax.set_title("Top 15 Features — Decision Tree")
    ax.invert_yaxis(); ax.grid(alpha=0.3, axis="x")
    save_and_show(fig, "bc_dt_importance")
    st.dataframe(imp.head(10).reset_index(drop=True).round(4), use_container_width=True)

    divider()
    secao("Cross-Validation cv=10")
    with st.spinner("Executando CV..."):
        cv_acc = cross_val_score(pipe_dt, X_bc, y_bc, cv=10, scoring="accuracy")
        cv_f1  = cross_val_score(pipe_dt, X_bc, y_bc, cv=10, scoring="f1_macro")
    cv1, cv2 = st.columns(2)
    cv1.metric("CV Accuracy", f"{cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
    cv2.metric("CV F1-macro", f"{cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

    divider()
    secao("Matriz de Confusão")
    cm_arr = confusion_matrix(yte, ypred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["benign","malignant"], yticklabels=["benign","malignant"],
                linewidths=1)
    ax.set_xlabel("Previsto"); ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão — Decision Tree")
    save_and_show(fig, "bc_dt_cm")

# ============================================================================
# RANDOM FOREST
# ============================================================================
elif pagina == "🌲  Random Forest":
    st.markdown('<h2>🌲 Random Forest — Ensemble com Bagging</h2>', unsafe_allow_html=True)
    divider()

    caixa("<b>O que é Random Forest?</b><br>"
          "Combina <b>múltiplas árvores de decisão</b> treinadas independentemente. "
          "Cada árvore é treinada com uma <b>amostra aleatória</b> dos dados "
          "(<b>bootstrap</b>) e considera apenas um <b>subconjunto aleatório de features</b> "
          "em cada divisão. A previsão final é por <b>votação majoritária</b>.")

    divider()
    secao("Bagging — Como Funciona?")
    cb1, cb2, cb3 = st.columns(3)
    with cb1:
        caixa("<b>Passo 1 — Bootstrap</b><br>"
              "Para cada árvore, sorteia uma amostra aleatória <b>com reposição</b> "
              "do conjunto de treino. Cada árvore vê um conjunto ligeiramente diferente.",
              cor="#f0f8ff", borda="#4C72B0")
    with cb2:
        caixa("<b>Passo 2 — Feature Sampling</b><br>"
              "Em cada divisão, considera apenas <b>√n_features</b> features "
              "aleatórias (classificação). Isso garante diversidade entre as árvores "
              "e reduz correlação.",
              cor="#f0f8ff", borda="#4C72B0")
    with cb3:
        caixa("<b>Passo 3 — Votação</b><br>"
              "Para classificação: cada árvore vota em uma classe. "
              "A classe com <b>maioria de votos</b> é a predição final. "
              "Para regressão: média das predições.",
              cor="#f0f8ff", borda="#4C72B0")

    divider()
    with st.spinner("Carregando Breast Cancer..."):
        df_bc, _ = carregar_breast_cancer()
    X_bc = df_bc.drop("target", axis=1); y_bc = df_bc["target"]
    Xtr, Xte, ytr, yte = train_test_split(
        X_bc, y_bc, test_size=0.2, stratify=y_bc, random_state=42)

    secao("Efeito do número de árvoras (n_estimators)")
    n_est = st.slider("n_estimators:", 5, 200, 50, 5, key="ne_rf")
    max_d_rf = st.slider("max_depth:", 2, 20, 10, 1, key="md_rf")

    pipe_rf = Pipeline([("rf", RandomForestClassifier(
        n_estimators=n_est, max_depth=max_d_rf, criterion="gini", random_state=42))])
    pipe_rf.fit(Xtr, ytr); ypred_rf = pipe_rf.predict(Xte)
    acc_rf = accuracy_score(yte, ypred_rf); f1_rf = f1_score(yte, ypred_rf, average="macro")

    cr1, cr2, cr3 = st.columns(3)
    cr1.metric("Accuracy", f"{acc_rf:.4f}"); cr2.metric("F1-macro", f"{f1_rf:.4f}")
    cr3.metric("n_estimators", str(n_est))

    secao("Relatório de Classificação")
    st.dataframe(pd.DataFrame(classification_report(yte,ypred_rf,output_dict=True)).T.round(4),
                 use_container_width=True)

    divider()
    secao("Curva: n_estimators vs Accuracy (Treino e Teste)")
    with st.spinner("Calculando curva..."):
        n_range = [5,10,20,30,50,75,100,150,200]
        accs_tr, accs_te = [], []
        for n in n_range:
            rf_n = RandomForestClassifier(n_estimators=n, random_state=42)
            rf_n.fit(Xtr, ytr)
            accs_tr.append(accuracy_score(ytr, rf_n.predict(Xtr)))
            accs_te.append(accuracy_score(yte, rf_n.predict(Xte)))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(n_range, accs_tr, "b-o", markersize=5, linewidth=2, label="Treino")
    ax.plot(n_range, accs_te, "r-s", markersize=5, linewidth=2, label="Teste")
    ax.set_xlabel("n_estimators"); ax.set_ylabel("Accuracy")
    ax.set_title("Random Forest — n_estimators vs Accuracy")
    ax.legend(); ax.grid(alpha=0.3)
    save_and_show(fig, "rf_n_estimators_curva")

    divider()
    secao("Feature Importance — Random Forest")
    rf_model = pipe_rf.named_steps["rf"]
    imp_rf = pd.DataFrame({
        "feature": X_bc.columns,
        "importance": rf_model.feature_importances_
    }).sort_values("importance", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp_rf["feature"][:15], imp_rf["importance"][:15], color="#38a169")
    ax.set_xlabel("Importância"); ax.set_title("Top 15 Features — Random Forest")
    ax.invert_yaxis(); ax.grid(alpha=0.3, axis="x")
    save_and_show(fig, "rf_importance")

    divider()
    secao("Cross-Validation cv=10")
    with st.spinner("Executando CV..."):
        cv_acc_rf = cross_val_score(pipe_rf, X_bc, y_bc, cv=10, scoring="accuracy")
        cv_f1_rf  = cross_val_score(pipe_rf, X_bc, y_bc, cv=10, scoring="f1_macro")
    ca, cf = st.columns(2)
    ca.metric("CV Accuracy", f"{cv_acc_rf.mean():.4f} ± {cv_acc_rf.std():.4f}")
    cf.metric("CV F1-macro", f"{cv_f1_rf.mean():.4f} ± {cv_f1_rf.std():.4f}")

    divider()
    with st.expander("Ver código Python — Random Forest"):
        code_block("""
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Random Forest NÃO precisa de StandardScaler
pipe_rf = Pipeline(steps=[
    ('rf', RandomForestClassifier(
        n_estimators=100,   # número de árvores
        criterion='gini',   # critério de impureza
        max_depth=None,     # None = cresce até pureza total
        random_state=42
    ))
])

pipe_rf.fit(X_train, y_train)
y_pred = pipe_rf.predict(X_test)

# Feature importance
rf = pipe_rf.named_steps['rf']
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).head(10)
""")

# ============================================================================
# XGBOOST
# ============================================================================
elif pagina == "⚡  XGBoost":
    st.markdown('<h2>⚡ XGBoost — Extreme Gradient Boosting</h2>', unsafe_allow_html=True)
    divider()

    caixa("<b>O que é XGBoost?</b><br>"
          "Ensemble de árvores construídas <b>sequencialmente</b>. Cada nova árvore "
          "corrige os <b>resíduos (erros)</b> do modelo atual — processo guiado pelo "
          "<b>gradiente da função de perda</b>. Mais preciso que RF na maioria dos "
          "problemas tabulares, porém mais sensível a hiperparâmetros.")

    divider()
    secao("Bagging vs Boosting")
    cc1, cc2 = st.columns(2)
    with cc1:
        caixa("<b>🌲 Random Forest (Bagging)</b><br>"
              "• Árvores treinadas em <b>paralelo</b><br>"
              "• Cada árvore independente<br>"
              "• Reduz <b>variância</b><br>"
              "• Robusto, difícil de overfittar<br>"
              "• Bom com poucos hiperparâmetros",
              cor="#f0f8ff", borda="#4C72B0")
    with cc2:
        caixa("<b>⚡ XGBoost (Boosting)</b><br>"
              "• Árvores treinadas em <b>sequência</b><br>"
              "• Cada árvore foca nos erros da anterior<br>"
              "• Reduz <b>viés e variância</b><br>"
              "• Mais propenso a overfitting<br>"
              "• Geralmente mais preciso, mais parâmetros",
              cor="#fff8e1", borda="#f59e0b")

    divider()

    if not XGBOOST_OK:
        caixa("<b>⚠️ XGBoost não instalado</b><br>"
              "Execute: <code>pip install xgboost</code> para habilitar esta seção.",
              cor="#fff0f0", borda="#e53e3e")
        with st.expander("Ver código Python — XGBoost"):
            code_block("""
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# XGBoost precisa do target como número (0, 1, 2...)
le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, stratify=y, random_state=42)

pipe_xgb = Pipeline(steps=[
    ('xgb', XGBClassifier(
        n_estimators=100,    # número de árvores
        max_depth=6,         # profundidade máxima
        learning_rate=0.1,   # taxa de aprendizado (eta)
        subsample=0.8,       # amostragem de linhas por árvore
        colsample_bytree=0.8,# amostragem de features por árvore
        random_state=42,
        eval_metric='logloss'
    ))
])

pipe_xgb.fit(X_train, y_train)
y_pred = pipe_xgb.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))
""")
    else:
        from xgboost import XGBClassifier

        with st.spinner("Carregando e treinando XGBoost..."):
            df_bc, _ = carregar_breast_cancer()
            X_bc = df_bc.drop("target", axis=1); y_bc = df_bc["target"]
            le = LabelEncoder(); y_enc = le.fit_transform(y_bc)
            Xtr, Xte, ytr_enc, yte_enc = train_test_split(
                X_bc, y_enc, test_size=0.2, stratify=y_bc, random_state=42)

            n_est_xgb = st.slider("n_estimators:", 10, 300, 100, 10, key="ne_xgb")
            lr_xgb    = st.select_slider("learning_rate:", [0.01, 0.05, 0.1, 0.2, 0.3], value=0.1, key="lr_xgb")
            md_xgb    = st.slider("max_depth:", 2, 10, 6, 1, key="md_xgb")

            pipe_xgb = Pipeline([("xgb", XGBClassifier(
                n_estimators=n_est_xgb, max_depth=md_xgb,
                learning_rate=lr_xgb, random_state=42,
                eval_metric="logloss", verbosity=0))])
            pipe_xgb.fit(Xtr, ytr_enc)
            ypred_xgb = pipe_xgb.predict(Xte)
            acc_xgb = accuracy_score(yte_enc, ypred_xgb)
            f1_xgb  = f1_score(yte_enc, ypred_xgb, average="macro")

        cx1, cx2, cx3 = st.columns(3)
        cx1.metric("Accuracy", f"{acc_xgb:.4f}")
        cx2.metric("F1-macro", f"{f1_xgb:.4f}")
        cx3.metric("n_estimators", str(n_est_xgb))

        secao("Relatório de Classificação")
        st.dataframe(pd.DataFrame(
            classification_report(yte_enc, ypred_xgb,
                                  target_names=le.classes_, output_dict=True)
        ).T.round(4), use_container_width=True)

        divider()
        secao("Feature Importance — XGBoost")
        xgb_model = pipe_xgb.named_steps["xgb"]
        imp_xgb = pd.DataFrame({
            "feature": X_bc.columns,
            "importance": xgb_model.feature_importances_
        }).sort_values("importance", ascending=False)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(imp_xgb["feature"][:15], imp_xgb["importance"][:15], color="#764ba2")
        ax.set_xlabel("Importância"); ax.set_title("Top 15 Features — XGBoost")
        ax.invert_yaxis(); ax.grid(alpha=0.3, axis="x")
        save_and_show(fig, "xgb_importance")

        divider()
        secao("Cross-Validation cv=10")
        with st.spinner("Executando CV..."):
            cv_acc_xgb = cross_val_score(pipe_xgb, X_bc, y_enc, cv=10, scoring="accuracy")
            cv_f1_xgb  = cross_val_score(pipe_xgb, X_bc, y_enc, cv=10, scoring="f1_macro")
        cxa, cxf = st.columns(2)
        cxa.metric("CV Accuracy", f"{cv_acc_xgb.mean():.4f} ± {cv_acc_xgb.std():.4f}")
        cxf.metric("CV F1-macro", f"{cv_f1_xgb.mean():.4f} ± {cv_f1_xgb.std():.4f}")

        divider()
        with st.expander("Ver código Python — XGBoost"):
            code_block("""
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# XGBoost exige target numérico (LabelEncoder)
le = LabelEncoder()
y_enc = le.fit_transform(y)

pipe_xgb = Pipeline(steps=[
    ('xgb', XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42
    ))
])

pipe_xgb.fit(X_train, y_train_enc)
y_pred = pipe_xgb.predict(X_test)

# Feature importance
xgb = pipe_xgb.named_steps['xgb']
imp = pd.Series(xgb.feature_importances_, index=X.columns)
imp.sort_values(ascending=False).head(10)
""")

# ============================================================================
# COMPARATIVO DE MODELOS
# ============================================================================
elif pagina == "📊  Comparativo de Modelos":
    st.markdown('<h2>📊 Comparativo — Decision Tree vs Random Forest vs XGBoost</h2>',
                unsafe_allow_html=True)
    divider()

    caixa("<b>Quando usar cada modelo?</b><br>"
          "• <b>Decision Tree</b>: quando interpretabilidade total é necessária "
          "(auditoria, regulação)<br>"
          "• <b>Random Forest</b>: quando você quer robustez sem ajustar muitos "
          "hiperparâmetros<br>"
          "• <b>XGBoost</b>: quando performance é a prioridade e você tem tempo "
          "para tunar hiperparâmetros")

    with st.spinner("Carregando dataset..."):
        df_bc, _ = carregar_breast_cancer()
    X_bc = df_bc.drop("target", axis=1); y_bc = df_bc["target"]
    le = LabelEncoder(); y_enc = le.fit_transform(y_bc)
    Xtr, Xte, ytr_enc, yte_enc = train_test_split(
        X_bc, y_enc, test_size=0.2, stratify=y_bc, random_state=42)

    modelos = [
        ("Decision Tree",  Pipeline([("dt", DecisionTreeClassifier(max_depth=10, random_state=42))])),
        ("Random Forest",  Pipeline([("rf", RandomForestClassifier(n_estimators=100, random_state=42))])),
    ]
    if XGBOOST_OK:
        from xgboost import XGBClassifier
        modelos.append(("XGBoost", Pipeline([("xgb", XGBClassifier(
            n_estimators=100, eval_metric="logloss", verbosity=0, random_state=42))])))

    with st.spinner("Treinando e comparando (CV=10)..."):
        resultados = []
        for nome, pipe_m in modelos:
            cv_acc = cross_val_score(pipe_m, X_bc, y_enc, cv=10, scoring="accuracy")
            cv_f1  = cross_val_score(pipe_m, X_bc, y_enc, cv=10, scoring="f1_macro")
            pipe_m.fit(Xtr, ytr_enc)
            acc_te = accuracy_score(yte_enc, pipe_m.predict(Xte))
            resultados.append({
                "Modelo": nome,
                "Acc (CV=10)": f"{cv_acc.mean():.4f} ± {cv_acc.std():.4f}",
                "F1 (CV=10)":  f"{cv_f1.mean():.4f} ± {cv_f1.std():.4f}",
                "Acc Teste":   f"{acc_te:.4f}",
            })

    st.dataframe(pd.DataFrame(resultados), use_container_width=True)

    nomes = [r["Modelo"] for r in resultados]
    accs  = [float(r["Acc (CV=10)"].split(" ±")[0]) for r in resultados]
    f1s   = [float(r["F1 (CV=10)"].split(" ±")[0]) for r in resultados]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(nomes))
    ax.bar(x-.2, accs, width=.35, label="Accuracy", color="#667eea", alpha=.9)
    ax.bar(x+.2, f1s,  width=.35, label="F1-macro",  color="#38a169", alpha=.9)
    for i,(a,f) in enumerate(zip(accs,f1s)):
        ax.text(i-.2, a+.002, f"{a:.3f}", ha="center", fontsize=10)
        ax.text(i+.2, f+.002, f"{f:.3f}", ha="center", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(nomes, fontsize=11)
    ax.set_ylabel("Métrica (CV=10)")
    ax.set_title("DT vs RF vs XGBoost — Breast Cancer (CV=10)")
    ax.set_ylim(0.88, 1.02); ax.legend(); ax.grid(alpha=.3, axis="y")
    save_and_show(fig, "comparativo_dt_rf_xgb")

    divider()
    secao("Interpretação")
    cm1, cm2, cm3 = st.columns(3)
    with cm1:
        caixa("<b>🌳 Decision Tree</b><br>"
              "✅ Interpretável — você pode ver cada regra<br>"
              "✅ Rápido para treinar<br>"
              "❌ Overfitting fácil<br>"
              "❌ Instável (pequenas mudanças nos dados mudam a árvore)")
    with cm2:
        caixa("<b>🌲 Random Forest</b><br>"
              "✅ Robusto — difícil de overfittar<br>"
              "✅ Pouco tuning necessário<br>"
              "✅ Feature importance confiável<br>"
              "❌ Menos interpretável (caixa preta)<br>"
              "❌ Mais lento que DT")
    with cm3:
        caixa("<b>⚡ XGBoost</b><br>"
              "✅ Geralmente mais preciso<br>"
              "✅ Regularização nativa<br>"
              "❌ Mais sensível a hiperparâmetros<br>"
              "❌ Mais lento para treinar<br>"
              "❌ Target precisa ser numérico")

    divider()
    st.markdown("""<div class="footer">
        <p style="margin:0;font-size:1.1rem;font-weight:700;">Machine Learning — Aula 06</p>
        <p style="margin:.3rem 0 0;font-size:.9rem;opacity:.8;">
        Decision Tree, Random Forest e XGBoost · Cláudio Ferreira Neves · SENAI/SC</p></div>""",
        unsafe_allow_html=True)
