"""
=============================================================================
Machine Learning — Aula {NN}
{Tema Completo}
Aplicação Streamlit interativa — material didático para iniciantes

Autor       : Cláudio Ferreira Neves
Cargo atual : Analista de BI — Save Co. | Jaraguá do Sul/SC
Docência    : Especialista de Ensino II — Análise de Dados | SENAI/SC
Certificação: DATA ANALYST CERTIFIED PROFESSIONAL © (DACP)
=============================================================================

INSTRUÇÕES DE USO:
  1. Substitua todos os campos entre chaves {} pelo conteúdo da aula.
  2. Crie uma função render_NOME() para cada seção da aula.
  3. Chame todas as funções render_*() dentro de main().
  4. Não use st.pyplot(fig) diretamente — use sempre save_and_show().
  5. Cada slider/input deve ter um key= único (ex: key="sl_grau_poly").
"""

import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Imports scikit-learn — adicione/remova conforme a aula
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression, LogisticRegression
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error

# ---------------------------------------------------------------------------
# OBRIGATÓRIO: backend Agg antes de qualquer plt — impede janelas externas
# ---------------------------------------------------------------------------
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Navegação entre aulas — ajuste os caminhos conforme necessário
# ---------------------------------------------------------------------------
PAGE_PORTAL       = "pages/Portal.py"
PAGE_AULA_ANTERIOR = "pages/Aula_{NN-1}.py"   # remova se for a primeira aula
PAGE_PROXIMA_AULA  = "pages/Aula_{NN+1}.py"   # remova se for a última aula

# ---------------------------------------------------------------------------
# Pasta de saída para salvar os gráficos gerados
# ---------------------------------------------------------------------------
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def save_and_show(fig: plt.Figure, filename: str) -> None:
    """Salva a figura em outputs/ com 150 DPI e exibe no Streamlit."""
    path = os.path.join(OUTPUTS_DIR, f"{filename}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    st.pyplot(fig)
    plt.close(fig)


# Adicione outras funções auxiliares reutilizáveis aqui
# Exemplo: plot_decision_boundary(), plot_confusion_matrix(), etc.


# ============================================================================
# SEÇÕES DA AULA — uma função por seção
# ============================================================================

def render_secao_1() -> None:
    """Seção 1 — {Nome da Seção}."""
    st.header("1. {Nome da Seção}")

    # -------------------------------------------------------------------
    # Explicação teórica
    # -------------------------------------------------------------------
    st.info("""
    **{Conceito principal}**

    {Explicação em 2–3 linhas. Use linguagem simples. Inclua analogia se útil.}
    """)

    # -------------------------------------------------------------------
    # Controles interativos (sidebar ou inline)
    # -------------------------------------------------------------------
    with st.sidebar:
        st.subheader("⚙️ Parâmetros — Seção 1")
        param_1 = st.slider(
            "{Label do controle}",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            key="s1_param_1",
            help="{Dica: o que esse parâmetro controla}",
        )

    # -------------------------------------------------------------------
    # Código / lógica da seção
    # -------------------------------------------------------------------
    np.random.seed(42)

    # {Lógica da seção — substitua pelos cálculos reais}
    X = np.random.randn(param_1, 1)
    y = 2 * X + np.random.randn(param_1, 1) * 0.5

    # -------------------------------------------------------------------
    # Visualização
    # -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X, y, alpha=0.6, color="steelblue", label="Dados")
    ax.set_xlabel("{Eixo X}")
    ax.set_ylabel("{Eixo Y}")
    ax.set_title("{Título do gráfico}")
    ax.legend()
    save_and_show(fig, "01_{descricao_do_grafico}")

    # -------------------------------------------------------------------
    # Resultado / métrica exibida ao aluno
    # -------------------------------------------------------------------
    st.success(f"**Resultado:** {'{métrica calculada}'}")


def render_secao_2() -> None:
    """Seção 2 — {Nome da Seção}."""
    st.header("2. {Nome da Seção}")

    st.info("""
    **{Conceito principal}**

    {Explicação em 2–3 linhas.}
    """)

    # {Controles, lógica e visualização da seção 2}


# Adicione render_secao_N() para cada seção adicional da aula


# ============================================================================
# MAIN — ponto de entrada da aplicação
# ============================================================================

def main() -> None:
    # -----------------------------------------------------------------------
    # Configuração da página — DEVE ser a primeira chamada Streamlit
    # -----------------------------------------------------------------------
    st.set_page_config(
        page_title="Aula {NN} — {Tema}",
        page_icon="{EMOJI}",
        layout="wide",
    )

    # -----------------------------------------------------------------------
    # Cabeçalho
    # -----------------------------------------------------------------------
    st.title("{EMOJI} Aula {NN} — {Tema Completo}")
    st.caption("Cláudio Ferreira Neves · Especialista em Ciência de Dados e IA")

    # -----------------------------------------------------------------------
    # Navegação — topo
    # -----------------------------------------------------------------------
    col_prev, col_portal, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("← Aula {NN-1}", use_container_width=True):
            st.switch_page(PAGE_AULA_ANTERIOR)
    with col_portal:
        if st.button("🏠 Portal", use_container_width=True):
            st.switch_page(PAGE_PORTAL)
    with col_next:
        if st.button("Aula {NN+1} →", use_container_width=True):
            st.switch_page(PAGE_PROXIMA_AULA)

    st.divider()

    # -----------------------------------------------------------------------
    # Objetivos da aula
    # -----------------------------------------------------------------------
    with st.expander("📋 O que você vai aprender nesta aula", expanded=False):
        st.markdown("""
        Ao final desta aula você será capaz de:

        - ✅ {Objetivo 1 — verbo no infinitivo: "Entender...", "Aplicar...", "Comparar..."}
        - ✅ {Objetivo 2}
        - ✅ {Objetivo 3}
        - ✅ {Objetivo 4}

        **Pré-requisitos:** {O que o aluno precisa saber antes desta aula}
        """)

    # -----------------------------------------------------------------------
    # Seções da aula — chame na ordem correta
    # -----------------------------------------------------------------------
    render_secao_1()
    st.divider()

    render_secao_2()
    st.divider()

    # render_secao_N()  ← adicione quantas seções forem necessárias

    # -----------------------------------------------------------------------
    # Navegação — rodapé (espelha o topo)
    # -----------------------------------------------------------------------
    st.divider()
    col_prev2, col_portal2, col_next2 = st.columns([1, 2, 1])
    with col_prev2:
        if st.button("← Aula {NN-1} ", use_container_width=True):  # key único: espaço extra
            st.switch_page(PAGE_AULA_ANTERIOR)
    with col_portal2:
        if st.button("🏠 Portal ", use_container_width=True):
            st.switch_page(PAGE_PORTAL)
    with col_next2:
        if st.button("Aula {NN+1} → ", use_container_width=True):
            st.switch_page(PAGE_PROXIMA_AULA)


if __name__ == "__main__":
    main()
