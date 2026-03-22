"""
=============================================================================
pages/Portal.py — Portal de Machine Learning
Página central de navegação entre as aulas do curso

Autor       : Cláudio Ferreira Neves
Cargo atual : Analista de BI — Save Co. | Jaraguá do Sul/SC
Docência    : Especialista de Ensino II — Análise de Dados | SENAI/SC
Certificação: DATA ANALYST CERTIFIED PROFESSIONAL © (DACP)

Como executar:
  streamlit run app.py
  (o portal e todas as aulas abrem automaticamente — sem múltiplos terminais)
=============================================================================
"""

import os
import streamlit as st

# ---------------------------------------------------------------------------
# Páginas das aulas (multi-page app)
# ---------------------------------------------------------------------------
PAGE_AULA_01 = "pages/Aula_01.py"
PAGE_AULA_02 = "pages/Aula_02.py"
PAGE_AULA_03 = "pages/Aula_03.py"
PAGE_AULA_04 = "pages/Aula_04.py"
PAGE_AULA_05 = "pages/Aula_05.py"
PAGE_AULA_06 = "pages/Aula_06.py"
PAGE_AULA_07 = "pages/Aula_07.py"
PAGE_AULA_08 = "pages/Aula_08.py"
PAGE_LANDING  = "app.py"
PAGE_ALUNO    = "pages/Aluno.py"

# ============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="Portal — Machine Learning",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CSS
# ============================================================================

st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
[data-testid="stSidebarNav"] { display: none; }

.sidebar-header {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    padding: 1.2rem; border-radius: 10px;
    text-align: center; margin-bottom: 1rem; color: white;
}

.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px; padding: 3rem 2rem 2.5rem;
    text-align: center; color: white; margin-bottom: 2rem;
}
.hero h1 { font-size: 2.8rem; font-weight: 800; margin: 0 0 0.5rem; }
.hero p  { font-size: 1.1rem; opacity: 0.85; margin: 0 auto; max-width: 680px; }
.hero .badge {
    display: inline-block; background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 20px; padding: 0.25rem 0.9rem;
    font-size: 0.85rem; margin: 1rem 0.3rem 0;
}

.aula-card {
    background: white; border-radius: 14px;
    border: 1px solid #e8ecf4;
    padding: 1.6rem 1.8rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.07);
    height: 100%;
    margin-bottom: 1rem;
}

.aula-number {
    font-size: 0.78rem; font-weight: 700; letter-spacing: 1.5px;
    text-transform: uppercase; color: #667eea; margin-bottom: 0.3rem;
}
.aula-title {
    font-size: 1.25rem; font-weight: 800; color: #1a1a2e;
    margin-bottom: 0.6rem; line-height: 1.3;
}
.aula-desc {
    font-size: 0.9rem; color: #555; margin-bottom: 1rem; line-height: 1.6;
}

.badge-disponivel {
    background: #d1fae5; color: #065f46;
    border-radius: 20px; padding: 0.2rem 0.8rem;
    font-size: 0.78rem; font-weight: 600;
    display: inline-block; margin-bottom: 1rem;
}

.topico {
    display: inline-block; background: #f0f4ff;
    border: 1px solid #c7d2fe; color: #3730a3;
    border-radius: 6px; padding: 0.15rem 0.6rem;
    font-size: 0.76rem; margin: 0.2rem 0.15rem 0 0;
}

.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px; padding: 1rem 1.5rem; color: white;
    text-align: center; margin-bottom: 0.5rem;
    box-shadow: 0 4px 15px rgba(102,126,234,0.35);
}
.metric-card h2 { margin: 0; font-size: 2rem; }
.metric-card p  { margin: 0; opacity: 0.85; font-size: 0.88rem; }

.tip-box {
    background: #fff8e1; border-left: 5px solid #f59e0b;
    padding: 1rem 1.5rem; border-radius: 0 10px 10px 0; margin: 0.8rem 0;
}
.info-box {
    background: #f0f4ff; border-left: 5px solid #667eea;
    padding: 1rem 1.5rem; border-radius: 0 10px 10px 0; margin: 0.8rem 0;
}
.success-box {
    background: #f0fff4; border-left: 5px solid #38a169;
    padding: 1rem 1.5rem; border-radius: 0 10px 10px 0; margin: 0.8rem 0;
}

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
        <h2 style="margin:0; font-size:1.3rem;">🎓 Machine Learning</h2>
        <p style="margin:0.4rem 0 0; font-size:0.8rem; opacity:0.8;">Portal de Navegação</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🗺️ Trilha completa")

    aulas_sidebar = [
        ("✅", "Aula 01", "Introdução ao ML"),
        ("✅", "Aula 02", "EDA"),
        ("✅", "Aula 03", "Regressão Linear e Polinomial"),
        ("✅", "Aula 04", "Regressão Logística e KNN"),
        ("✅", "Aula 05", "Naive Bayes e SVM"),
        ("✅", "Aula 06", "Decision Tree e Random Forest"),
        ("✅", "Aula 07", "K-Means e PCA"),
        ("✅", "Aula 08", "Comparação de Modelos"),
    ]

    for icon, num, titulo in aulas_sidebar:
        st.markdown(
            f"<div style='background:#f0fff4; border-radius:8px; padding:0.5rem 0.75rem; "
            f"margin-bottom:0.3rem; border-left:3px solid #38a169;'>"
            f"<span style='font-size:0.85rem; font-weight:600; color:#1a1a2e;'>{icon} {num}</span><br>"
            f"<span style='font-size:0.76rem; color:#555;'>{titulo}</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("""
    <div style='font-size:0.75rem; color:#888; text-align:center; margin-top:1rem;'>
        Cláudio Ferreira Neves<br>Especialista em Ciência de Dados e IA
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("📓 Área do Aluno", use_container_width=True, key="sidebar_btn_aluno"):
        st.switch_page(PAGE_ALUNO)
    st.page_link("app.py", label="← Voltar à Apresentação", use_container_width=True)


# ============================================================================
# AUTOR
# ============================================================================

st.markdown(
    "<p style='text-align:center; font-size:0.95rem; color:#667eea; "
    "font-weight:600; margin-bottom:0;'>"
    "Cláudio Ferreira Neves &nbsp;·&nbsp; Especialista em Ciência de Dados e IA"
    "</p>",
    unsafe_allow_html=True,
)

# ============================================================================
# HERO
# ============================================================================

st.markdown("""
<div class="hero">
    <h1>🎓 Portal de Machine Learning</h1>
    <p>Curso completo de Machine Learning — 8 aulas interativas com Python e scikit-learn</p>
    <div>
        <span class="badge">📚 8 Aulas</span>
        <span class="badge">🐍 Python</span>
        <span class="badge">🤖 scikit-learn</span>
        <span class="badge">📊 EDA</span>
        <span class="badge">🔬 CRISP-DM</span>
        <span class="badge">✅ Completo</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# MÉTRICAS
# ============================================================================

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-card"><h2>8</h2><p>Aulas disponíveis</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card"><h2>✅</h2><p>Curso completo</p></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card"><h2>🐍</h2><p>Python + Streamlit</p></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-card"><h2>🔬</h2><p>CRISP-DM</p></div>', unsafe_allow_html=True)

st.markdown("---")


# ============================================================================
# AULAS — LINHA 1 (01 a 04)
# ============================================================================

st.markdown("## 📖 Aulas disponíveis")
st.markdown(
    "<p style='color:#555; margin-top:-0.5rem; margin-bottom:1.5rem;'>"
    "Clique em <b>Abrir aula</b> para iniciar. "
    "A navegação entre aulas acontece pelas abas — sem abrir novos terminais.</p>",
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4, gap="medium")

with col1:
    st.markdown("""
    <div class="aula-card">
        <div class="aula-number">Aula 01</div>
        <div class="aula-title">🤖 Introdução ao Machine Learning</div>
        <div><span class="badge-disponivel">✅ Disponível</span></div>
        <div class="aula-desc">
            Fundamentos de ML: como modelos aprendem padrões e como avaliamos se aprenderam bem.
            Da teoria à prática com scikit-learn.
        </div>
        <div style="margin-bottom:1rem;">
            <span class="topico">Regressão Linear</span>
            <span class="topico">Classificação</span>
            <span class="topico">Train/Test Split</span>
            <span class="topico">Dataset IRIS</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🚀 Abrir Aula 01", use_container_width=True, type="primary", key="btn_aula01"):
        st.switch_page(PAGE_AULA_01)

with col2:
    st.markdown("""
    <div class="aula-card">
        <div class="aula-number">Aula 02</div>
        <div class="aula-title">🔍 Análise Exploratória de Dados (EDA)</div>
        <div><span class="badge-disponivel">✅ Disponível</span></div>
        <div class="aula-desc">
            Antes de modelar, entenda seus dados. Inspeção, limpeza, visualização
            e insights com CRISP-DM e Palmer Penguins.
        </div>
        <div style="margin-bottom:1rem;">
            <span class="topico">CRISP-DM</span>
            <span class="topico">Valores Ausentes</span>
            <span class="topico">Correlação</span>
            <span class="topico">Outliers (IQR)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🚀 Abrir Aula 02", use_container_width=True, type="primary", key="btn_aula02"):
        st.switch_page(PAGE_AULA_02)

with col3:
    st.markdown("""
    <div class="aula-card">
        <div class="aula-number">Aula 03</div>
        <div class="aula-title">📈 Regressão Linear e Polinomial</div>
        <div><span class="badge-disponivel">✅ Disponível</span></div>
        <div class="aula-desc">
            Preveja números contínuos com regressão linear simples, múltipla
            e polinomial. Gradiente descendente e datasets reais.
        </div>
        <div style="margin-bottom:1rem;">
            <span class="topico">Linear Simples</span>
            <span class="topico">Linear Múltipla</span>
            <span class="topico">Polinomial</span>
            <span class="topico">Pipeline</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🚀 Abrir Aula 03", use_container_width=True, type="primary", key="btn_aula03"):
        st.switch_page(PAGE_AULA_03)

with col4:
    st.markdown("""
    <div class="aula-card">
        <div class="aula-number">Aula 04</div>
        <div class="aula-title">🔮 Regressão Logística e KNN</div>
        <div><span class="badge-disponivel">✅ Disponível</span></div>
        <div class="aula-desc">
            Classificação com Regressão Logística e KNN. Penguins, Titanic,
            GridSearchCV e cross-validation estratificada.
        </div>
        <div style="margin-bottom:1rem;">
            <span class="topico">Regressão Logística</span>
            <span class="topico">KNN</span>
            <span class="topico">GridSearchCV</span>
            <span class="topico">Cross-Validation</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🚀 Abrir Aula 04", use_container_width=True, type="primary", key="btn_aula04"):
        st.switch_page(PAGE_AULA_04)


st.markdown("---")


# ============================================================================
# AULAS — LINHA 2 (05 a 08)
# ============================================================================

col5, col6, col7, col8 = st.columns(4, gap="medium")

with col5:
    st.markdown("""
    <div class="aula-card">
        <div class="aula-number">Aula 05</div>
        <div class="aula-title">🧠 Naive Bayes e SVM</div>
        <div><span class="badge-disponivel">✅ Disponível</span></div>
        <div class="aula-desc">
            Dois classificadores com abordagens opostas: probabilístico (GNB)
            e geométrico (SVM). Kernels, GridSearch e dataset Wine.
        </div>
        <div style="margin-bottom:1rem;">
            <span class="topico">GaussianNB</span>
            <span class="topico">SVM (RBF/Linear)</span>
            <span class="topico">Kernels</span>
            <span class="topico">Dataset Wine</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🚀 Abrir Aula 05", use_container_width=True, type="primary", key="btn_aula05"):
        st.switch_page(PAGE_AULA_05)

with col6:
    st.markdown("""
    <div class="aula-card">
        <div class="aula-number">Aula 06</div>
        <div class="aula-title">🌳 Decision Tree e Random Forest</div>
        <div><span class="badge-disponivel">✅ Disponível</span></div>
        <div class="aula-desc">
            Da árvore simples ao ensemble poderoso. Bagging vs Boosting,
            Feature Importance e XGBoost no Breast Cancer.
        </div>
        <div style="margin-bottom:1rem;">
            <span class="topico">Decision Tree</span>
            <span class="topico">Random Forest</span>
            <span class="topico">XGBoost</span>
            <span class="topico">Feature Importance</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🚀 Abrir Aula 06", use_container_width=True, type="primary", key="btn_aula06"):
        st.switch_page(PAGE_AULA_06)

with col7:
    st.markdown("""
    <div class="aula-card">
        <div class="aula-number">Aula 07</div>
        <div class="aula-title">🔵 K-Means e PCA</div>
        <div><span class="badge-disponivel">✅ Disponível</span></div>
        <div class="aula-desc">
            Aprendizado não supervisionado. Segmentação de clientes com K-Means,
            Elbow, Silhouette e redução de dimensionalidade com PCA.
        </div>
        <div style="margin-bottom:1rem;">
            <span class="topico">K-Means</span>
            <span class="topico">Elbow Method</span>
            <span class="topico">PCA</span>
            <span class="topico">Mall Customers</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🚀 Abrir Aula 07", use_container_width=True, type="primary", key="btn_aula07"):
        st.switch_page(PAGE_AULA_07)

with col8:
    st.markdown("""
    <div class="aula-card">
        <div class="aula-number">Aula 08</div>
        <div class="aula-title">🧬 Comparação de Modelos</div>
        <div><span class="badge-disponivel">✅ Disponível</span></div>
        <div class="aula-desc">
            Aula final: compare todos os modelos no diagnóstico de Alzheimer,
            construa um Voting Ensemble e salve o modelo com joblib.
        </div>
        <div style="margin-bottom:1rem;">
            <span class="topico">Comparativo CV=10</span>
            <span class="topico">VotingClassifier</span>
            <span class="topico">joblib</span>
            <span class="topico">Alzheimer Dataset</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🚀 Abrir Aula 08", use_container_width=True, type="primary", key="btn_aula08"):
        st.switch_page(PAGE_AULA_08)


st.markdown("---")


# ============================================================================
# TRILHA COMPLETA
# ============================================================================

st.markdown("## 🗺️ Trilha de aprendizado")

trilha = [
    ("✅", "Aula 01", "Introdução ao Machine Learning",
     "Regressão, classificação, train/test split e pipeline com scikit-learn.",
     "CRISP-DM: Entendimento do Negócio"),
    ("✅", "Aula 02", "Análise Exploratória de Dados (EDA)",
     "Inspeção, limpeza, univariada, multivariada e tratamento de outliers.",
     "CRISP-DM: Entendimento e Preparação dos Dados"),
    ("✅", "Aula 03", "Regressão Linear e Polinomial",
     "Linear simples, múltipla, polinomial, gradiente descendente.",
     "CRISP-DM: Modelagem — Regressão"),
    ("✅", "Aula 04", "Regressão Logística e KNN",
     "Classificação binária e multi-classe, distâncias, fronteiras de decisão.",
     "CRISP-DM: Modelagem — Classificação"),
    ("✅", "Aula 05", "Naive Bayes e SVM",
     "Classificação probabilística (GNB) e geométrica (SVM). Kernels e GridSearch.",
     "CRISP-DM: Modelagem — Classificação Avançada"),
    ("✅", "Aula 06", "Decision Tree e Random Forest",
     "Árvores, ensembles (bagging e boosting), feature importance.",
     "CRISP-DM: Modelagem — Ensemble"),
    ("✅", "Aula 07", "K-Means e PCA",
     "Clusterização não supervisionada e redução de dimensionalidade.",
     "CRISP-DM: Modelagem — Aprendizado Não Supervisionado"),
    ("✅", "Aula 08", "Comparação de Modelos",
     "Comparativo completo, VotingClassifier, joblib e deploy.",
     "CRISP-DM: Avaliação e Implantação"),
]

for icon, num, titulo, desc, fase in trilha:
    st.markdown(f"""
    <div style="background:#f0fff4; border-left:4px solid #38a169; border-radius:0 10px 10px 0;
                padding:0.8rem 1.2rem; margin-bottom:0.5rem; display:flex; gap:1rem; align-items:flex-start;">
        <span style="font-size:1.4rem; flex-shrink:0;">{icon}</span>
        <div style="flex:1;">
            <span style="font-size:0.75rem; font-weight:700; letter-spacing:1px;
                         text-transform:uppercase; color:#065f46;">{num}</span>
            <h4 style="margin:0.1rem 0 0.2rem; font-size:0.95rem; color:#1a1a2e;">{titulo}</h4>
            <p style="margin:0 0 0.2rem; font-size:0.83rem; color:#555;">{desc}</p>
            <span style="font-size:0.73rem; color:#888;">📍 {fase}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")


# ============================================================================
# COMO EXECUTAR
# ============================================================================

st.markdown("## ⚙️ Como executar as aulas")

col_a, col_b = st.columns(2, gap="large")

with col_a:
    st.markdown("""
    <div class="info-box">
        <b>Um único comando inicia todo o projeto:</b><br><br>
        <code>streamlit run app.py</code><br><br>
        O portal e todas as aulas abrem automaticamente. Não é necessário abrir múltiplos terminais.
    </div>
    """, unsafe_allow_html=True)

with col_b:
    st.markdown("""
    <div class="tip-box">
        💡 <b>Arquitetura multi-page Streamlit:</b><br><br>
        A pasta <code>pages/</code> registra cada aula como uma página separada. A navegação entre aulas acontece pelas abas no topo de cada página — sem abrir novos terminais.
    </div>
    <div class="success-box">
        <b>✅ Instalar dependências</b><br><br>
        <code>pip install streamlit numpy pandas matplotlib seaborn scikit-learn xgboost joblib</code>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# ÁREA DO ALUNO — CARD DE ACESSO
# ============================================================================

st.markdown("""
<div style="background: linear-gradient(135deg, #065f46 0%, #059669 60%, #0f766e 100%);
            border-radius: 16px; padding: 2rem 2.5rem; color: white;
            display: flex; align-items: center; gap: 2rem; margin-bottom: 1rem;">
    <div style="font-size: 3rem; flex-shrink: 0;">📓</div>
    <div style="flex: 1;">
        <h2 style="margin: 0 0 0.4rem; font-size: 1.6rem; font-weight: 800;">Área do Aluno</h2>
        <p style="margin: 0; opacity: 0.9; font-size: 0.95rem; max-width: 560px;">
            Notebooks com exercícios para praticar enquanto acompanha as aulas.
            Baixe, abra no Jupyter ou VS Code e complete os exercícios marcados com ✍️.
        </p>
        <p style="margin: 0.6rem 0 0; font-size: 0.82rem; opacity: 0.75;">
            ✅ Aula 01 disponível &nbsp;·&nbsp; 🔒 Aulas 02–08 em breve
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

if st.button("📓 Ir para a Área do Aluno →", use_container_width=False, type="primary", key="btn_aluno_main"):
    st.switch_page(PAGE_ALUNO)

st.markdown("---")


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("""
<div class="footer">
    <p style="margin:0; font-size:1rem; font-weight:600; opacity:0.9;">
        🎓 Machine Learning — Curso Completo
    </p>
    <p style="margin:0.4rem 0 0; font-size:0.85rem; opacity:0.7;">
        Material de autoria de Cláudio Ferreira Neves &nbsp;·&nbsp; SENAI/SC &nbsp;·&nbsp; 8 Aulas
    </p>
</div>
""", unsafe_allow_html=True)
