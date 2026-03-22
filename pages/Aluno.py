"""
=============================================================================
pages/Aluno.py — Área do Aluno
Material para prática: notebooks com exercícios em branco por aula

Autor       : Cláudio Ferreira Neves
Cargo atual : Analista de BI — Save Co. | Jaraguá do Sul/SC
Docência    : Especialista de Ensino II — Análise de Dados | SENAI/SC

Como executar:
  streamlit run app.py
=============================================================================
"""

import os
import streamlit as st

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
PAGE_PORTAL = "pages/Portal.py"

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="Área do Aluno — Machine Learning",
    page_icon="📓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
[data-testid="stSidebar"]    { display: none !important; }
[data-testid="stSidebarNav"] { display: none !important; }

.hero-aluno {
    background: linear-gradient(135deg, #065f46 0%, #059669 60%, #0f766e 100%);
    border-radius: 20px; padding: 3rem 2.5rem 2.5rem;
    text-align: center; color: white; margin-bottom: 2rem;
}
.hero-aluno h1 { font-size: 2.4rem; font-weight: 800; margin: 0 0 0.5rem; }
.hero-aluno p  { font-size: 1.05rem; opacity: 0.88; margin: 0 auto; max-width: 640px; }

.nb-card {
    background: white; border-radius: 14px;
    border: 1px solid #e8ecf4;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.07);
    height: 100%; margin-bottom: 1rem;
}
.nb-card-num  { font-size: 0.75rem; font-weight: 700; letter-spacing: 1.2px;
                text-transform: uppercase; color: #059669; margin-bottom: 0.3rem; }
.nb-card-title{ font-size: 1rem; font-weight: 800; color: #1a1a2e; margin-bottom: 0.5rem; }
.nb-card-desc { font-size: 0.85rem; color: #555; line-height: 1.6; margin-bottom: 1rem; }

.badge-ok     { background: #d1fae5; color: #065f46; border-radius: 20px;
                padding: 0.2rem 0.8rem; font-size: 0.75rem; font-weight: 600;
                display: inline-block; margin-bottom: 0.8rem; }
.badge-soon   { background: #f3f4f6; color: #9ca3af; border-radius: 20px;
                padding: 0.2rem 0.8rem; font-size: 0.75rem; font-weight: 600;
                display: inline-block; margin-bottom: 0.8rem; }

.how-card {
    background: #f0fff4; border-left: 4px solid #059669;
    border-radius: 0 12px 12px 0; padding: 1.2rem 1.5rem; margin-bottom: 0.8rem;
}
.how-card h4 { margin: 0 0 0.4rem; font-size: 0.95rem; color: #065f46; }
.how-card p  { margin: 0; font-size: 0.88rem; color: #374151; line-height: 1.6; }

.footer-aluno {
    background: #1a1a2e; color: white; border-radius: 12px;
    padding: 1.5rem 2rem; text-align: center; margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Navegação — botão voltar ao portal
# ---------------------------------------------------------------------------
if st.button("← Voltar ao Portal", key="btn_back_top"):
    st.switch_page(PAGE_PORTAL)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-aluno">
    <h1>📓 Área do Aluno</h1>
    <p>Notebooks com exercícios para você praticar enquanto acompanha as aulas.
       Baixe, abra no Jupyter ou VS Code e complete os exercícios marcados com ✍️.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Como usar
# ---------------------------------------------------------------------------
st.markdown("## 🛠️ Como usar os notebooks")
c1, c2, c3 = st.columns(3, gap="medium")
with c1:
    st.markdown("""
    <div class="how-card">
        <h4>1. Baixe o notebook</h4>
        <p>Clique no botão da aula correspondente abaixo para baixar o arquivo <code>.ipynb</code>.</p>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="how-card">
        <h4>2. Abra no Jupyter ou VS Code</h4>
        <p>Instale o Jupyter Notebook ou use a extensão Python do VS Code para abrir o arquivo.</p>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="how-card">
        <h4>3. Complete os exercícios ✍️</h4>
        <p>Células marcadas com ✍️ estão em branco para você preencher. Siga as dicas no código.</p>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Notebooks disponíveis
# ---------------------------------------------------------------------------
st.markdown("## 📚 Notebooks por aula")
st.markdown(
    "<p style='color:#555; margin-top:-0.5rem; margin-bottom:1.5rem;'>"
    "✅ Disponível para download &nbsp;·&nbsp; 🔒 Em breve</p>",
    unsafe_allow_html=True,
)

_AULAS = [
    {
        "num": "Aula 01", "emoji": "🤖",
        "titulo": "Introdução ao Machine Learning",
        "desc": "Regressão Linear, Regressão Polinomial, Classificação com IRIS e Train/Test Split.",
        "file": "aula_01/Aula_01_Introducao_ao_ML_(aluno).ipynb",
        "exercicios": 7,
    },
    {
        "num": "Aula 02", "emoji": "🔍",
        "titulo": "Análise Exploratória de Dados",
        "desc": "CRISP-DM, análise univariada e multivariada, tratamento de outliers com IQR.",
        "file": "aula_02/Aula_02_Analise_Exploratoria_(aluno).ipynb",
        "exercicios": 6,
    },
    {
        "num": "Aula 03", "emoji": "📈",
        "titulo": "Regressão Linear e Polinomial",
        "desc": "Linear simples, múltipla, polinomial, Pipeline e gradiente descendente.",
        "file": "aula_03/Aula_03_Regressao_Linear_Polinomial_(aluno).ipynb",
        "exercicios": 7,
    },
    {
        "num": "Aula 04", "emoji": "🔮",
        "titulo": "Regressão Logística e KNN",
        "desc": "Classificação binária e multi-classe, GridSearchCV e cross-validation.",
        "file": "aula_04/Aula_04_Regressao_Logistica_KNN_(aluno).ipynb",
        "exercicios": 6,
    },
    {
        "num": "Aula 05", "emoji": "⚡",
        "titulo": "Naive Bayes e SVM",
        "desc": "Classificação probabilística (GNB) e geométrica (SVM). Kernels e GridSearch.",
        "file": None,
    },
    {
        "num": "Aula 06", "emoji": "🌳",
        "titulo": "Decision Tree e Random Forest",
        "desc": "Árvores, ensembles, bagging, boosting, XGBoost e Feature Importance.",
        "file": None,
    },
    {
        "num": "Aula 07", "emoji": "🎯",
        "titulo": "K-Means e PCA",
        "desc": "Clusterização não supervisionada, Elbow Method e redução de dimensionalidade.",
        "file": None,
    },
    {
        "num": "Aula 08", "emoji": "🧬",
        "titulo": "Comparação de Modelos",
        "desc": "Comparativo com cv=10, VotingClassifier, joblib e dataset Alzheimer.",
        "file": None,
    },
]

# Renderiza em grid 4 colunas, 2 linhas
for row_start in range(0, len(_AULAS), 4):
    cols = st.columns(4, gap="medium")
    for col, aula in zip(cols, _AULAS[row_start:row_start + 4]):
        nb_path = os.path.join(_ROOT, aula["file"]) if aula.get("file") else None
        disponivel = nb_path and os.path.exists(nb_path)
        with col:
            badge = '<span class="badge-ok">✅ Disponível</span>' if disponivel else '<span class="badge-soon">🔒 Em breve</span>'
            exerc = f'<p style="font-size:0.78rem; color:#059669; margin:0 0 0.8rem;"><b>{aula["exercicios"]} exercícios</b> para completar</p>' if disponivel else ""
            st.markdown(f"""
            <div class="nb-card">
                <div class="nb-card-num">{aula["num"]}</div>
                <div class="nb-card-title">{aula["emoji"]} {aula["titulo"]}</div>
                {badge}
                <div class="nb-card-desc">{aula["desc"]}</div>
                {exerc}
            </div>""", unsafe_allow_html=True)
            if disponivel:
                with open(nb_path, "rb") as _f:
                    st.download_button(
                        label=f"⬇️ Baixar — {aula['num']}",
                        data=_f,
                        file_name=os.path.basename(aula["file"]),
                        mime="application/json",
                        use_container_width=True,
                        key=f"dl_{aula['num'].replace(' ', '_')}",
                    )
            else:
                st.button(
                    "🔒 Em breve",
                    disabled=True,
                    use_container_width=True,
                    key=f"soon_{aula['num'].replace(' ', '_')}",
                )

st.markdown("---")

# ---------------------------------------------------------------------------
# Dica final
# ---------------------------------------------------------------------------
st.info(
    "💡 **Dica:** Acompanhe a aula interativa no Streamlit e use o notebook para praticar em paralelo. "
    "Os exercícios seguem a mesma sequência do conteúdo apresentado na aula."
)

# ---------------------------------------------------------------------------
# Botão voltar
# ---------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
if st.button("← Voltar ao Portal", key="btn_back_bottom"):
    st.switch_page(PAGE_PORTAL)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("""
<div class="footer-aluno">
    <p style="margin:0; font-size:0.9rem; font-weight:600; opacity:0.9;">
        📓 Área do Aluno — Machine Learning
    </p>
    <p style="margin:0.3rem 0 0; font-size:0.8rem; opacity:0.65;">
        Cláudio Ferreira Neves &nbsp;·&nbsp; Especialista em Ciência de Dados e IA &nbsp;·&nbsp; SENAI/SC
    </p>
</div>
""", unsafe_allow_html=True)
