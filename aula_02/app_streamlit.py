"""
=============================================================================
Machine Learning — Aula 02
Análise Exploratória de Dados (EDA)
Aplicação Streamlit interativa — material didático para iniciantes

Autor       : Cláudio Ferreira Neves
Cargo atual : Analista de BI — Save Co. | Jaraguá do Sul/SC
Docência    : Especialista de Ensino II — Análise de Dados | SENAI/SC
Certificação: DATA ANALYST CERTIFIED PROFESSIONAL © (DACP)
=============================================================================
"""

import os
import math
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
PAGE_AULA_01 = "pages/Aula_01.py"
PAGE_AULA_03 = "pages/Aula_03.py"

import seaborn as sns
import streamlit as st
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


# ============================================================================
# CARREGAMENTO E LIMPEZA DOS DADOS (cache para performance)
# ============================================================================

@st.cache_data
def carregar_dados() -> pd.DataFrame:
    """Carrega o dataset Palmer Penguins via seaborn."""
    df = sns.load_dataset("penguins")
    df = df.rename(columns={
        "species": "espécie",
        "island": "ilha",
        "bill_length_mm": "comprimento_bico_mm",
        "bill_depth_mm": "profundidade_bico_mm",
        "flipper_length_mm": "comprimento_nadadeira_mm",
        "body_mass_g": "massa_corporal_g",
        "sex": "sexo",
        "year": "ano",
    })
    df["sexo"] = df["sexo"].map({"male": "macho", "female": "fêmea"})
    return df


@st.cache_data
def dados_limpos() -> pd.DataFrame:
    """Retorna o dataset após remoção de valores ausentes e duplicatas."""
    df = carregar_dados().copy()
    df = df.dropna()
    df = df.drop_duplicates()
    return df.reset_index(drop=True)


@st.cache_data
def dados_sem_outliers() -> pd.DataFrame:
    """
    Remove outliers pelo método IQR aplicado por espécie,
    sobre as colunas numéricas do dataset limpo.
    """
    df = dados_limpos().copy()
    num_cols = df.select_dtypes(include="float64").columns.tolist()

    def iqr_mask(grupo: pd.DataFrame, col: str) -> pd.Series:
        Q1 = grupo[col].quantile(0.25)
        Q3 = grupo[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return (grupo[col] >= lower) & (grupo[col] <= upper)

    mask = pd.Series(True, index=df.index)
    for col in num_cols:
        col_mask = df.groupby("espécie", group_keys=False).apply(
            lambda g: iqr_mask(g, col)
        )
        mask = mask & col_mask

    return df[mask].reset_index(drop=True)


# ============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="EDA — Palmer Penguins",
    page_icon="🐧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS — visual premium (mesmo padrão da aula_01)
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
        <h2 style="margin:0; font-size:1.3rem;">📊 CRISP-DM</h2>
        <p style="margin:0.4rem 0 0; font-size:0.8rem; opacity:0.8;">Análise Exploratória de Dados</p>
    </div>
    """, unsafe_allow_html=True)

    pagina = st.radio(
        "Navegação",
        options=[
            "🏠  Início",
            "📋  CRISP-DM",
            "🔍  Estrutura dos Dados",
            "🧹  Limpeza de Dados",
            "📊  Análise Univariada",
            "🔗  Análise Multivariada",
            "🎯  Tratamento de Outliers",
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
tab_nav(2)  # replace CURRENT_NUMBER with the correct int

st.markdown(
    "<p style='text-align:center; font-size:0.95rem; color:#667eea; "
    "font-weight:600; margin-bottom:0;'>"
    "Cláudio Ferreira Neves &nbsp;·&nbsp; Especialista em Ciência de Dados e IA"
    "</p>",
    unsafe_allow_html=True,
)

_nav_l, _nav_m, _nav_r = st.columns([1.2, 4, 1.2])
with _nav_l:
    if st.button("← Aula 01", use_container_width=True, key="nav_prev"):
        st.switch_page(PAGE_AULA_01)
with _nav_m:
    if st.button("🏠 Portal", use_container_width=True, key="nav_portal"):
        st.switch_page(PAGE_PORTAL)
with _nav_r:
    if st.button("Aula 03 →", use_container_width=True, key="nav_next"):
        st.switch_page(PAGE_AULA_03)

# ============================================================================
# PÁGINA: INÍCIO
# ============================================================================

if pagina == "🏠  Início":

    st.markdown("""
    <div style='text-align:center; padding: 2rem 0 1rem;'>
        <h1 style='font-size:2.8rem; font-weight:800; color:#1a1a2e; margin-bottom:0.3rem;'>
            🐧 Análise Exploratória de Dados
        </h1>
        <p style='font-size:1.15rem; color:#555; max-width:720px; margin:0 auto;'>
            Explore o dataset Palmer Penguins passo a passo, seguindo o fluxo
            CRISP-DM: da compreensão dos dados à entrega de um conjunto limpo
            e pronto para modelagem.
        </p>
    </div>
    """, unsafe_allow_html=True)

    divider()

    # --- Métricas ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><h2>7</h2><p>Seções</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h2>344</h2><p>Amostras</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h2>🐧</h2><p>Espécies</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><h2>📊</h2><p>EDA</p></div>', unsafe_allow_html=True)

    divider()

    st.markdown("## 📚 O que você vai aprender")
    col_a, col_b = st.columns(2)

    with col_a:
        caixa_conceito("""
        <b>📋 CRISP-DM</b><br>
        O processo padrão de mineração de dados que guia todo projeto de ciência de dados
        do negócio até a entrega em produção.
        """)
        caixa_conceito("""
        <b>🔍 Estrutura dos Dados</b><br>
        Como inspecionar um DataFrame: shape, dtypes, valores únicos, nulos e duplicatas.
        """)
        caixa_conceito("""
        <b>🧹 Limpeza de Dados</b><br>
        Estratégias para tratar valores ausentes (drop vs impute) e remover registros duplicados.
        """)

    with col_b:
        caixa_conceito("""
        <b>📊 Análise Univariada</b><br>
        Estatísticas descritivas, distribuições, histogramas e boxplots para
        entender cada variável isoladamente.
        """)
        caixa_conceito("""
        <b>🔗 Análise Multivariada</b><br>
        Correlações, pairplot e gráficos agrupados para revelar relações entre variáveis.
        """)
        caixa_conceito("""
        <b>🎯 Tratamento de Outliers</b><br>
        Método IQR aplicado por grupo para identificar e remover valores extremos com precisão.
        """)

    divider()

    secao("🛠️ Bibliotecas utilizadas nesta aula")
    col1, col2, col3, col4 = st.columns(4)
    libs = [
        ("🐼", "Pandas", "Manipulação e análise de dados tabulares"),
        ("🔢", "NumPy", "Operações numéricas e cálculos estatísticos"),
        ("📉", "Matplotlib", "Visualizações base e customizações gráficas"),
        ("🎨", "Seaborn", "Gráficos estatísticos de alto nível"),
    ]
    for col, (icon, name, desc) in zip([col1, col2, col3, col4], libs):
        with col:
            st.markdown(
                f'<div class="concept-box" style="text-align:center;">'
                f'<span style="font-size:1.8rem;">{icon}</span><br>'
                f'<b>{name}</b><br><small>{desc}</small></div>',
                unsafe_allow_html=True,
            )

    divider()
    caixa_dica("""
    <b>Como usar este app:</b> navegue pelas seções na barra lateral esquerda em ordem crescente
    — cada seção constrói sobre o conhecimento da anterior.
    Todos os gráficos gerados são salvos automaticamente na pasta <code>outputs/</code>.
    """)

    st.markdown("""
    <div class="footer">
        <p style="margin:0; font-size:1rem; font-weight:600;">
            Machine Learning — Aula 02 · Análise Exploratória de Dados
        </p>
        <p style="margin:0.3rem 0 0; font-size:0.85rem; opacity:0.7;">
            Dataset: Palmer Penguins · Horst AM, Hill AP, Gorman KB (2020)
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# PÁGINA: CRISP-DM
# ============================================================================

elif pagina == "📋  CRISP-DM":

    st.markdown("""
    <div style='text-align:center; padding: 1.5rem 0 0.5rem;'>
        <h1 style='font-size:2.4rem; font-weight:800; color:#1a1a2e;'>
            📋 CRISP-DM
        </h1>
        <p style='font-size:1.05rem; color:#555;'>
            Cross Industry Standard Process for Data Mining — o mapa do projeto de dados
        </p>
    </div>
    """, unsafe_allow_html=True)

    divider()

    caixa_conceito("""
    <b>O que é CRISP-DM?</b><br>
    CRISP-DM é um processo iterativo e cíclico que descreve as etapas para conduzir
    projetos de mineração de dados e machine learning de forma estruturada, independente
    do setor ou tecnologia utilizada. Foi desenvolvido em 1996 e ainda é o padrão mais
    adotado na indústria.
    """)

    divider()

    secao("🔄 As 6 Fases do CRISP-DM")

    fases = [
        (
            "1️⃣ Business Understanding — Entendimento do Negócio",
            "#e8f4fd",
            "#2196f3",
            """
            Antes de tocar nos dados, precisamos entender <b>o problema a resolver</b>.<br><br>
            Perguntas-chave: Qual é o objetivo do projeto? Qual métrica define sucesso?
            Quais são as restrições de tempo e custo? No caso do dataset Penguins,
            nosso objetivo é compreender as características morfológicas de três espécies
            de pinguins da Antártida para, futuramente, construir um classificador.
            """,
        ),
        (
            "2️⃣ Data Understanding — Compreensão dos Dados",
            "#f0f4ff",
            "#667eea",
            """
            Coletamos os dados e os exploramos: quantas linhas? Que colunas existem?
            Há valores ausentes? Quais são as distribuições? Esta fase é o núcleo
            desta aula — cobrida pelas seções <b>Estrutura dos Dados</b> e
            <b>Análise Univariada / Multivariada</b>.
            """,
        ),
        (
            "3️⃣ Data Preparation — Preparação dos Dados",
            "#f0fff4",
            "#38a169",
            """
            Limpamos, transformamos e estruturamos os dados para que os algoritmos
            possam consumi-los. Inclui: tratar nulos, remover duplicatas, eliminar
            outliers, encodar variáveis categóricas, normalizar escalas.
            Coberto pelas seções <b>Limpeza de Dados</b> e <b>Tratamento de Outliers</b>.
            """,
        ),
        (
            "4️⃣ Modeling — Modelagem",
            "#fff8e1",
            "#f59e0b",
            """
            Selecionamos e treinamos algoritmos de ML sobre os dados preparados.
            Ajustamos hiperparâmetros e comparamos resultados.
            <i>(Foco das próximas aulas.)</i>
            """,
        ),
        (
            "5️⃣ Evaluation — Avaliação",
            "#fff0f0",
            "#e53e3e",
            """
            Avaliamos se o modelo realmente atende ao objetivo de negócio definido
            na fase 1. Usamos métricas como acurácia, F1-Score, RMSE, etc.
            <i>(Foco das próximas aulas.)</i>
            """,
        ),
        (
            "6️⃣ Deployment — Implantação",
            "#f5f0ff",
            "#7c3aed",
            """
            Entregamos o modelo ao ambiente de produção: API REST, dashboard,
            relatório automatizado ou pipeline de batch. O ciclo reinicia quando
            o mundo muda e o modelo precisa ser retreinado.
            <i>(Foco das próximas aulas.)</i>
            """,
        ),
    ]

    for titulo, bg, border, texto in fases:
        st.markdown(
            f'<div style="background:{bg}; border-left:5px solid {border}; '
            f'padding:1rem 1.5rem; border-radius:0 10px 10px 0; margin:0.8rem 0;">'
            f'<b>{titulo}</b><br>{texto}</div>',
            unsafe_allow_html=True,
        )

    divider()

    st.markdown("""
    <div class="explain-box">
        <b>🎯 Foco desta aula — Fases 2 e 3</b><br>
        Nesta aula cobrimos as fases de <b>Data Understanding</b> e <b>Data Preparation</b>,
        que juntas formam a base de qualquer projeto de Machine Learning bem-sucedido.
        Dados mal compreendidos ou sujos levam invariavelmente a modelos ruins,
        independentemente do algoritmo utilizado: <i>"garbage in, garbage out"</i>.
    </div>
    """, unsafe_allow_html=True)

    divider()

    secao("📦 Importações utilizadas nesta aula")
    with st.expander("Ver código de importação", expanded=True):
        code_block("""
import pandas as pd          # DataFrames e manipulação de dados tabulares
import numpy as np           # Operações numéricas vetorizadas
import matplotlib.pyplot as plt  # Criação de gráficos base
import matplotlib            # Configurações do backend
import seaborn as sns        # Gráficos estatísticos de alto nível

# Usar backend não-interativo (obrigatório no Streamlit)
matplotlib.use("Agg")

# Carregar o dataset Palmer Penguins diretamente do seaborn
df = sns.load_dataset("penguins")
print(df.shape)  # (344, 7)
""")


# ============================================================================
# PÁGINA: ESTRUTURA DOS DADOS
# ============================================================================

elif pagina == "🔍  Estrutura dos Dados":

    st.markdown("""
    <div style='text-align:center; padding: 1.5rem 0 0.5rem;'>
        <h1 style='font-size:2.4rem; font-weight:800; color:#1a1a2e;'>
            🔍 Estrutura dos Dados
        </h1>
        <p style='font-size:1.05rem; color:#555;'>
            A primeira coisa a fazer ao receber um dataset: <b>conhecê-lo profundamente</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    divider()

    df = carregar_dados()

    caixa_conceito("""
    <b>Por que explorar a estrutura?</b><br>
    Antes de qualquer análise ou modelagem, precisamos responder: Quantas linhas e colunas?
    Quais tipos de dados? Há valores ausentes? Há duplicatas? Essas respostas definem
    todo o trabalho de preparação que virá a seguir.
    """)

    divider()

    # --- Seletor interativo ---
    opcao = st.radio(
        "O que deseja visualizar?",
        options=[
            "Primeiras linhas (head)",
            "Dimensões (shape)",
            "Informações gerais (info)",
            "Tipos de dados (dtypes)",
            "Colunas",
            "Valores únicos por coluna",
            "Valores ausentes",
            "Duplicatas",
        ],
        horizontal=False,
    )

    divider()

    if opcao == "Primeiras linhas (head)":
        secao("📄 df.head(n)")
        caixa_conceito("""
        <b>df.head(n)</b> retorna as primeiras <i>n</i> linhas do DataFrame.
        É o primeiro comando que todo cientista de dados executa para ter uma
        visão rápida dos dados: ver os nomes das colunas, os tipos de valores
        e identificar possíveis problemas imediatamente.
        """)
        n = st.slider("Número de linhas a exibir", min_value=5, max_value=20, value=5, step=1)
        st.dataframe(df.head(n), use_container_width=True)

    elif opcao == "Dimensões (shape)":
        secao("📐 df.shape")
        caixa_conceito("""
        <b>df.shape</b> retorna uma tupla <code>(linhas, colunas)</code>.
        Saber o tamanho do dataset é fundamental: datasets pequenos permitem
        inspeção manual; datasets grandes exigem amostragem antes de visualizar.
        """)
        rows, cols = df.shape
        c1, c2 = st.columns(2)
        c1.metric("Linhas (amostras)", rows)
        c2.metric("Colunas (features)", cols)
        st.code(f"df.shape  →  {df.shape}", language="python")

    elif opcao == "Informações gerais (info)":
        secao("ℹ️ df.info()")
        caixa_conceito("""
        <b>df.info()</b> é um dos comandos mais informativos: mostra o nome de cada coluna,
        quantos valores não-nulos existem e o dtype de cada uma. A discrepância entre
        o total de linhas e o count de não-nulos revela diretamente onde estão os nulos.
        """)
        import io
        buf = io.StringIO()
        df.info(buf=buf)
        st.text(buf.getvalue())

    elif opcao == "Tipos de dados (dtypes)":
        secao("🔤 df.dtypes")
        caixa_conceito("""
        <b>df.dtypes</b> mostra o tipo de cada coluna. Os principais tipos são:
        <code>float64</code> (número decimal), <code>int64</code> (inteiro),
        <code>object</code> (texto/categoria) e <code>category</code>.
        Algoritmos de ML geralmente exigem que todas as entradas sejam numéricas —
        colunas <code>object</code> precisarão de encoding.
        """)
        st.dataframe(
            pd.DataFrame({"Coluna": df.columns, "Tipo": df.dtypes.values}),
            use_container_width=True,
        )

    elif opcao == "Colunas":
        secao("📋 df.columns")
        caixa_conceito("""
        <b>df.columns</b> lista os nomes de todas as colunas do DataFrame.
        No dataset Penguins temos 7 colunas: 3 categóricas (espécie, ilha, sexo)
        e 4 numéricas (comprimento_bico_mm, profundidade_bico_mm, comprimento_nadadeira_mm, massa_corporal_g),
        além do ano de coleta.
        """)
        for i, col in enumerate(df.columns, 1):
            st.markdown(f"**{i}.** `{col}`")

    elif opcao == "Valores únicos por coluna":
        secao("🔢 Valores únicos por coluna")
        caixa_conceito("""
        Verificar os valores únicos por coluna nos ajuda a entender a cardinalidade:
        variáveis com poucos valores únicos são candidatas a variáveis categóricas;
        variáveis com muitos valores únicos são geralmente numéricas ou identificadores.
        """)
        for col in df.columns:
            n_unique = df[col].nunique()
            unique_vals = df[col].unique()
            with st.expander(f"**{col}** — {n_unique} valores únicos"):
                st.write(unique_vals)

    elif opcao == "Valores ausentes":
        secao("❓ Valores ausentes — df.isnull().sum()")
        caixa_conceito("""
        <b>df.isnull().sum()</b> conta quantos valores ausentes (NaN) existem por coluna.
        Valores ausentes são inevitáveis em dados reais e precisam ser tratados antes
        da modelagem. As estratégias principais são: <b>remover</b> as linhas/colunas
        afetadas ou <b>imputar</b> (preencher com média, mediana, moda ou modelo).
        """)
        nulos = df.isnull().sum().reset_index()
        nulos.columns = ["Coluna", "Nulos"]
        nulos["Percentual (%)"] = (nulos["Nulos"] / len(df) * 100).round(2)

        st.dataframe(nulos, use_container_width=True)

        # Gráfico de barras dos valores ausentes
        fig, ax = plt.subplots(figsize=(8, 4))
        colunas_com_nulos = nulos[nulos["Nulos"] > 0]
        if len(colunas_com_nulos) > 0:
            ax.bar(
                colunas_com_nulos["Coluna"],
                colunas_com_nulos["Nulos"],
                color="#667eea",
                edgecolor="white",
            )
            ax.set_title("Quantidade de Valores Ausentes por Coluna", fontsize=13, fontweight="bold")
            ax.set_xlabel("Coluna")
            ax.set_ylabel("Nulos")
            for bar in ax.patches:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    str(int(bar.get_height())),
                    ha="center", va="bottom", fontsize=11,
                )
            fig.tight_layout()
            save_and_show(fig, "nulos_por_coluna")
        else:
            st.success("Nenhum valor ausente encontrado!")
            plt.close(fig)

    elif opcao == "Duplicatas":
        secao("🔁 Duplicatas — df.duplicated().sum()")
        caixa_conceito("""
        Linhas duplicadas distorcem estatísticas e modelos — um mesmo pinguim
        contado duas vezes infla sua importância artificialmente.
        <b>df.duplicated().sum()</b> conta quantas linhas são cópias exatas de outra linha.
        """)
        n_dup = df.duplicated().sum()
        if n_dup == 0:
            st.success(f"Nenhuma linha duplicada encontrada. (Total: {len(df)} linhas)")
        else:
            st.warning(f"{n_dup} linha(s) duplicada(s) encontrada(s)!")
            st.dataframe(df[df.duplicated()], use_container_width=True)

    divider()

    with st.expander("📝 Ver código desta seção"):
        code_block("""
import seaborn as sns
import pandas as pd
import io

# Carregar o dataset
df = sns.load_dataset("penguins")

# Primeiras linhas
print(df.head(5))

# Dimensões
print(f"Shape: {df.shape}")         # (344, 7)

# Informações gerais (dtypes + nulos)
buf = io.StringIO()
df.info(buf=buf)
print(buf.getvalue())

# Tipos de dados
print(df.dtypes)

# Nomes das colunas
print(df.columns.tolist())

# Valores únicos por coluna
for col in df.columns:
    print(f"{col}: {df[col].nunique()} únicos → {df[col].unique()}")

# Valores ausentes por coluna
print(df.isnull().sum())

# Percentual de nulos
print((df.isnull().sum() / len(df) * 100).round(2))

# Duplicatas
print(f"Linhas duplicadas: {df.duplicated().sum()}")
""")


# ============================================================================
# PÁGINA: LIMPEZA DE DADOS
# ============================================================================

elif pagina == "🧹  Limpeza de Dados":

    st.markdown("""
    <div style='text-align:center; padding: 1.5rem 0 0.5rem;'>
        <h1 style='font-size:2.4rem; font-weight:800; color:#1a1a2e;'>
            🧹 Limpeza de Dados
        </h1>
        <p style='font-size:1.05rem; color:#555;'>
            Dados sujos geram modelos ruins. Limpar é obrigatório, não opcional.
        </p>
    </div>
    """, unsafe_allow_html=True)

    divider()

    df_raw = carregar_dados()
    df_clean = dados_limpos()

    caixa_conceito("""
    <b>Por que limpar os dados?</b><br>
    A maioria dos algoritmos de Machine Learning não consegue lidar com valores ausentes
    ou duplicatas — eles causam erros ou distorcem os resultados.
    A limpeza garante que nosso conjunto de dados seja <b>consistente, completo e confiável</b>
    antes de qualquer análise ou modelagem.
    """)

    divider()

    # --- Antes e depois ---
    secao("📊 Antes × Depois da Limpeza")

    col1, col2, col3 = st.columns(3)
    col1.metric("Linhas originais", len(df_raw))
    col2.metric("Linhas após limpeza", len(df_clean))
    col3.metric("Linhas removidas", len(df_raw) - len(df_clean), delta=f"-{len(df_raw) - len(df_clean)}", delta_color="inverse")

    divider()

    # --- Nulos ---
    secao("❓ Passo 1 — Remoção de valores ausentes")

    caixa_conceito("""
    <b>dropna() vs impute — quando usar cada um?</b><br>
    <ul>
      <li><b>dropna()</b>: simples e seguro quando os dados ausentes são aleatórios
          e representam uma pequena fração (&lt; 5%). Perdemos informação, mas garantimos
          qualidade.</li>
      <li><b>Imputação</b>: preferível quando os nulos não são aleatórios ou quando
          perderíamos muitos dados. Preenchemos com média, mediana, moda ou um
          modelo preditivo.</li>
    </ul>
    No Penguins temos apenas 11 linhas com nulos (≈3,2%), então <code>dropna()</code>
    é a escolha correta.
    """)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Nulos ANTES:**")
        st.dataframe(
            df_raw.isnull().sum().rename("Nulos").to_frame(),
            use_container_width=True,
        )
    with col_b:
        st.markdown("**Nulos DEPOIS (dropna):**")
        st.dataframe(
            df_clean.isnull().sum().rename("Nulos").to_frame(),
            use_container_width=True,
        )

    divider()

    # --- Duplicatas ---
    secao("🔁 Passo 2 — Remoção de duplicatas")

    caixa_conceito("""
    <b>Por que remover duplicatas?</b><br>
    Linhas duplicadas fazem com que certas amostras tenham peso desproporcional
    no treinamento. Isso pode enviesar o modelo para "memorizar" esses registros
    repetidos em vez de aprender padrões genuínos.
    """)

    n_dup_raw = df_raw.duplicated().sum()
    n_dup_clean = df_clean.duplicated().sum()

    col1, col2 = st.columns(2)
    col1.metric("Duplicatas antes", n_dup_raw)
    col2.metric("Duplicatas depois", n_dup_clean)

    if n_dup_raw == 0:
        st.success("Nenhuma duplicata encontrada no dataset original.")

    divider()

    caixa_atencao("""
    <b>Atenção — perda de dados:</b> Ao remover as 11 linhas com valores ausentes,
    passamos de 344 para 333 amostras (perda de ≈3,2%). Esse trade-off é aceitável
    aqui, mas em datasets menores ou com muitos nulos, a imputação seria obrigatória
    para não comprometer a representatividade dos dados.
    """)

    divider()

    secao("👀 Amostra do dataset limpo")
    st.dataframe(df_clean.head(10), use_container_width=True)

    divider()

    with st.expander("📝 Ver código desta seção"):
        code_block("""
import seaborn as sns
import pandas as pd

# Carregar dataset original
df = sns.load_dataset("penguins")
print(f"Shape original: {df.shape}")   # (344, 7)
print(f"Nulos:\\n{df.isnull().sum()}")
print(f"Duplicatas: {df.duplicated().sum()}")

# Passo 1 — Remover linhas com valores ausentes
df_clean = df.dropna()
print(f"Shape após dropna: {df_clean.shape}")  # (333, 7)

# Passo 2 — Remover duplicatas (se houver)
df_clean = df_clean.drop_duplicates()

# Resetar índice para evitar lacunas na numeração
df_clean = df_clean.reset_index(drop=True)

print(f"Shape final: {df_clean.shape}")
print(f"Nulos restantes: {df_clean.isnull().sum().sum()}")
print(f"Duplicatas restantes: {df_clean.duplicated().sum()}")
""")


# ============================================================================
# PÁGINA: ANÁLISE UNIVARIADA
# ============================================================================

elif pagina == "📊  Análise Univariada":

    st.markdown("""
    <div style='text-align:center; padding: 1.5rem 0 0.5rem;'>
        <h1 style='font-size:2.4rem; font-weight:800; color:#1a1a2e;'>
            📊 Análise Univariada
        </h1>
        <p style='font-size:1.05rem; color:#555;'>
            Entenda cada variável isoladamente antes de cruzar informações.
        </p>
    </div>
    """, unsafe_allow_html=True)

    divider()

    df = dados_limpos()
    num_cols = df.select_dtypes(include="float64").columns.tolist()
    cat_cols = ["espécie", "ilha", "sexo"]

    caixa_conceito("""
    <b>O que é Análise Univariada?</b><br>
    Analisamos <b>uma variável por vez</b> para entender sua distribuição, tendência
    central, dispersão e a presença de valores extremos. É o ponto de partida obrigatório
    antes de cruzar variáveis — você não pode interpretar relações sem antes entender
    cada peça individualmente.
    """)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📐 Estatísticas Descritivas",
        "🏷️ Variáveis Categóricas",
        "📊 Histogramas",
        "📦 Boxplots",
    ])

    # ---- Tab 1: Estatísticas Descritivas ----
    with tab1:
        secao("📐 df.describe()")
        caixa_conceito("""
        <b>O que significa cada métrica do describe()?</b>
        <ul>
          <li><b>count</b>: número de valores não-nulos — útil para detectar nulos</li>
          <li><b>mean</b>: média aritmética — sensível a outliers</li>
          <li><b>std</b>: desvio padrão — mede a dispersão dos dados em torno da média</li>
          <li><b>min / max</b>: valores extremos — candidatos a outliers</li>
          <li><b>25% (Q1)</b>: 25% dos dados estão abaixo deste valor</li>
          <li><b>50% (Q2 / mediana)</b>: valor central — robusto a outliers</li>
          <li><b>75% (Q3)</b>: 75% dos dados estão abaixo deste valor</li>
        </ul>
        A diferença entre <b>Q3 - Q1</b> é chamada de <b>IQR (Intervalo Interquartil)</b>
        e é a base para a detecção de outliers.
        """)
        st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)

        divider()
        caixa_dica("""
        Compare a <b>média</b> e a <b>mediana (50%)</b> de cada coluna:
        quando estão próximas, a distribuição é aproximadamente simétrica.
        Quando diferem muito, a distribuição é assimétrica (skewed) e a mediana
        é mais representativa como medida central.
        """)

    # ---- Tab 2: Variáveis Categóricas ----
    with tab2:
        secao("🏷️ Análise de Variáveis Categóricas")
        caixa_conceito("""
        Variáveis categóricas não têm média ou desvio padrão — usamos
        <b>contagens</b> e <b>proporções</b> para entendê-las.
        O <code>value_counts()</code> mostra quantas vezes cada categoria aparece,
        revelando se o dataset é <b>balanceado</b> (importante para classificação).
        """)

        col_cat = st.radio(
            "Selecione a variável categórica:",
            options=cat_cols + ["ano"],
            horizontal=True,
        )

        vc = df[col_cat].value_counts().reset_index()
        vc.columns = [col_cat, "Contagem"]
        vc["Percentual (%)"] = (vc["Contagem"] / len(df) * 100).round(1)

        col_l, col_r = st.columns([1, 2])
        with col_l:
            st.dataframe(vc, use_container_width=True)

        with col_r:
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = sns.color_palette("husl", n_colors=len(vc))
            bars = ax.bar(vc[col_cat].astype(str), vc["Contagem"], color=colors, edgecolor="white")
            ax.set_title(f"Distribuição de '{col_cat}'", fontsize=13, fontweight="bold")
            ax.set_xlabel(col_cat.capitalize())
            ax.set_ylabel("Contagem")
            for bar, pct in zip(bars, vc["Percentual (%)"]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{pct}%",
                    ha="center", va="bottom", fontsize=10,
                )
            fig.tight_layout()
            save_and_show(fig, f"cat_{col_cat}")

    # ---- Tab 3: Histogramas ----
    with tab3:
        secao("📊 Histogramas")
        caixa_conceito("""
        O <b>histograma</b> divide os valores de uma variável contínua em intervalos (bins)
        e conta quantas observações caem em cada intervalo. Ele revela:
        <ul>
          <li>Se a distribuição é <b>unimodal</b> (um pico) ou <b>bimodal</b> (dois picos)</li>
          <li>Se é <b>simétrica</b>, assimétrica à direita (<i>right-skewed</i>) ou à esquerda</li>
          <li>A presença de <b>outliers</b> (barras isoladas nas extremidades)</li>
        </ul>
        """)

        col_hist = st.select_slider(
            "Selecione a variável numérica:",
            options=num_cols,
        )

        n_default = max(1, int(math.sqrt(len(df))))
        n_bins = st.slider("Número de bins", min_value=1, max_value=50, value=n_default)
        kde_on = st.checkbox("Sobrepor curva KDE (densidade)", value=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(
            data=df,
            x=col_hist,
            bins=n_bins,
            kde=kde_on,
            color="#667eea",
            edgecolor="white",
            ax=ax,
        )
        ax.set_title(f"Histograma — {col_hist}", fontsize=13, fontweight="bold")
        ax.set_xlabel(col_hist)
        ax.set_ylabel("Frequência")
        fig.tight_layout()
        save_and_show(fig, f"hist_{col_hist}_{n_bins}bins")

        caixa_dica("""
        A regra de <b>Sturges</b> sugere usar √n bins como ponto de partida,
        onde n é o número de observações. Para 333 amostras, isso resulta em
        aproximadamente 18 bins — mas o ideal é sempre ajustar visualmente.
        """)

    # ---- Tab 4: Boxplots ----
    with tab4:
        secao("📦 Boxplots")
        caixa_conceito("""
        O <b>boxplot</b> (diagrama de caixa) resume a distribuição em 5 números:
        <ul>
          <li><b>Linha central</b>: mediana (Q2 — 50% dos dados)</li>
          <li><b>Caixa</b>: do Q1 (25%) ao Q3 (75%) — contém 50% central dos dados</li>
          <li><b>Bigodes</b>: se estendem até 1,5 × IQR além da caixa</li>
          <li><b>Pontos isolados</b>: outliers — valores além dos bigodes</li>
        </ul>
        """)

        col_box = st.selectbox("Selecione a variável para boxplot:", options=num_cols)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.boxplot(
            df[col_box].dropna(),
            vert=True,
            patch_artist=True,
            boxprops=dict(facecolor="#667eea", color="#1a1a2e"),
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(color="#1a1a2e"),
            capprops=dict(color="#1a1a2e"),
            flierprops=dict(marker="o", color="#e53e3e", markersize=5),
        )
        ax.set_title(f"Boxplot — {col_box}", fontsize=13, fontweight="bold")
        ax.set_ylabel(col_box)
        ax.set_xticks([])
        fig.tight_layout()
        save_and_show(fig, f"boxplot_{col_box}")

        # Mostrar Q1, Q2, Q3, IQR
        q1 = df[col_box].quantile(0.25)
        q2 = df[col_box].quantile(0.50)
        q3 = df[col_box].quantile(0.75)
        iqr = q3 - q1
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Q1 (25%)", f"{q1:.2f}")
        c2.metric("Mediana (Q2)", f"{q2:.2f}")
        c3.metric("Q3 (75%)", f"{q3:.2f}")
        c4.metric("IQR", f"{iqr:.2f}")

    divider()
    with st.expander("📝 Ver código desta seção"):
        code_block("""
import seaborn as sns
import matplotlib.pyplot as plt
import math

df = sns.load_dataset("penguins").dropna().reset_index(drop=True)
num_cols = df.select_dtypes(include="float64").columns.tolist()

# --- Estatísticas descritivas ---
print(df.describe())

# --- Variáveis categóricas ---
print(df["espécie"].value_counts())
print((df["espécie"].value_counts() / len(df) * 100).round(1))

# --- Histograma com KDE ---
fig, ax = plt.subplots(figsize=(8, 4))
n_bins = int(math.sqrt(len(df)))   # Regra de Sturges
sns.histplot(data=df, x="comprimento_bico_mm", bins=n_bins, kde=True,
             color="#667eea", edgecolor="white", ax=ax)
ax.set_title("Histograma — comprimento_bico_mm")
plt.tight_layout()
plt.show()

# --- Boxplot ---
q1, q2, q3 = df["comprimento_bico_mm"].quantile([0.25, 0.50, 0.75])
iqr = q3 - q1
print(f"Q1={q1:.2f}  Q2={q2:.2f}  Q3={q3:.2f}  IQR={iqr:.2f}")
print(f"Outliers abaixo de: {q1 - 1.5*iqr:.2f}")
print(f"Outliers acima de: {q3 + 1.5*iqr:.2f}")

fig, ax = plt.subplots(figsize=(5, 5))
ax.boxplot(df["comprimento_bico_mm"], patch_artist=True)
ax.set_title("Boxplot — comprimento_bico_mm")
plt.tight_layout()
plt.show()
""")


# ============================================================================
# PÁGINA: ANÁLISE MULTIVARIADA
# ============================================================================

elif pagina == "🔗  Análise Multivariada":

    st.markdown("""
    <div style='text-align:center; padding: 1.5rem 0 0.5rem;'>
        <h1 style='font-size:2.4rem; font-weight:800; color:#1a1a2e;'>
            🔗 Análise Multivariada
        </h1>
        <p style='font-size:1.05rem; color:#555;'>
            Revele padrões ocultos cruzando duas ou mais variáveis simultaneamente.
        </p>
    </div>
    """, unsafe_allow_html=True)

    divider()

    df = dados_limpos()
    num_cols = df.select_dtypes(include="float64").columns.tolist()

    caixa_conceito("""
    <b>Por que analisar múltiplas variáveis juntas?</b><br>
    A análise univariada não revela <b>relações entre variáveis</b>. Por exemplo,
    duas espécies de pinguins podem ter comprimentos de bico similares, mas quando
    cruzamos com a profundidade do bico, a separação fica clara. A análise multivariada
    é o que transforma observações brutas em <b>insights acionáveis</b>.
    """)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Histogramas por Categoria",
        "📦 Boxplots por Categoria",
        "🌡️ Correlação",
        "🔭 Pairplot",
    ])

    # ---- Tab 1: Histogramas por categoria ----
    with tab1:
        secao("📊 Histogramas por Categoria")
        caixa_conceito("""
        Sobrepor histogramas de grupos diferentes no mesmo eixo é uma forma poderosa
        de comparar distribuições. Quando as distribuições de grupos distintos
        <b>se separam</b>, a variável é um bom discriminador — útil para classificação.
        """)

        col_num1 = st.selectbox("Variável numérica:", options=num_cols, key="h_num")
        grupo1 = st.radio("Agrupar por:", options=["espécie", "sexo"], horizontal=True, key="h_grp")

        fig, ax = plt.subplots(figsize=(9, 4))
        sns.histplot(
            data=df,
            x=col_num1,
            hue=grupo1,
            kde=True,
            element="step",
            palette="husl",
            ax=ax,
        )
        ax.set_title(f"Distribuição de '{col_num1}' por {grupo1}", fontsize=13, fontweight="bold")
        ax.set_xlabel(col_num1)
        ax.set_ylabel("Frequência")
        fig.tight_layout()
        save_and_show(fig, f"hist_cat_{col_num1}_{grupo1}")

    # ---- Tab 2: Boxplots por categoria ----
    with tab2:
        secao("📦 Boxplots por Categoria")
        caixa_conceito("""
        O boxplot agrupado compara mediana, dispersão e outliers entre categorias
        de forma compacta. É ideal para responder: "A medida do bico difere
        significativamente entre espécies?" Uma diferença clara nas medianas
        sugere que a variável é discriminativa.
        """)

        col_num2 = st.selectbox("Variável numérica:", options=num_cols, key="b_num")
        col_cat2 = st.selectbox("Agrupar por:", options=["espécie", "ilha", "sexo"], key="b_cat")

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(
            data=df,
            x=col_cat2,
            y=col_num2,
            palette="husl",
            width=0.5,
            ax=ax,
        )
        ax.set_title(f"{col_num2} por {col_cat2}", fontsize=13, fontweight="bold")
        ax.set_xlabel(col_cat2.capitalize())
        ax.set_ylabel(col_num2)
        fig.tight_layout()
        save_and_show(fig, f"boxcat_{col_num2}_{col_cat2}")

    # ---- Tab 3: Correlação ----
    with tab3:
        secao("🌡️ Mapa de Correlação (Heatmap)")
        caixa_conceito("""
        O <b>coeficiente de correlação de Pearson (r)</b> mede a relação linear entre
        duas variáveis numéricas, variando de -1 a +1:
        <ul>
          <li><b>r ≈ +1</b>: correlação positiva forte — ambas crescem juntas</li>
          <li><b>r ≈ 0</b>: sem correlação linear</li>
          <li><b>r ≈ -1</b>: correlação negativa forte — uma cresce enquanto a outra cai</li>
        </ul>
        O heatmap exibe todas as correlações de uma vez, facilitando a identificação
        de variáveis redundantes (alta correlação entre features pode prejudicar alguns modelos).
        """)

        corr = df[num_cols].corr()

        fig, ax = plt.subplots(figsize=(7, 5))
        mask = np.triu(np.ones_like(corr, dtype=bool))  # Apenas triângulo inferior
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            linewidths=0.5,
            linecolor="white",
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title("Matriz de Correlação — Variáveis Numéricas", fontsize=13, fontweight="bold")
        fig.tight_layout()
        save_and_show(fig, "correlacao_heatmap")

        divider()
        caixa_conceito("""
        <b>Principais achados no dataset Penguins:</b>
        <ul>
          <li><b>comprimento_nadadeira_mm × massa_corporal_g</b>: correlação fortemente positiva (~0.87)
              — pinguins com nadadeiras maiores são mais pesados.</li>
          <li><b>comprimento_bico_mm × comprimento_nadadeira_mm</b>: correlação positiva moderada (~0.66)
              — bicos mais longos tendem a ocorrer em pinguins maiores.</li>
          <li><b>profundidade_bico_mm × comprimento_nadadeira_mm</b>: correlação negativa (~-0.58)
              — profundidade do bico tende a ser menor em pinguins maiores.</li>
        </ul>
        """)

    # ---- Tab 4: Pairplot ----
    with tab4:
        secao("🔭 Pairplot — Visão Geral Completa")
        caixa_conceito("""
        O <b>pairplot</b> (ou scatter matrix) exibe gráficos de dispersão para todos
        os pares de variáveis numéricas, com histogramas ou KDE na diagonal.
        É a visão mais completa de um dataset de uma só vez — porém pode ser lento
        para datasets grandes. Clique no botão abaixo para gerar.
        """)

        cor_grp = st.radio("Colorir por:", options=["espécie", "sexo"], horizontal=True, key="pp_grp")
        gerar = st.button("🔭 Gerar Pairplot (pode demorar alguns segundos)")

        if gerar:
            with st.spinner("Gerando pairplot..."):
                fig = sns.pairplot(
                    df,
                    hue=cor_grp,
                    vars=num_cols,
                    diag_kind="kde",
                    plot_kws={"alpha": 0.6, "s": 25},
                    palette="husl",
                )
                fig.fig.suptitle(
                    f"Pairplot — colorido por {cor_grp}",
                    y=1.02, fontsize=14, fontweight="bold",
                )
                path = os.path.join(OUTPUTS_DIR, "pairplot.png")
                fig.fig.savefig(path, dpi=120, bbox_inches="tight")
                st.pyplot(fig.fig)
                plt.close("all")
            st.success("Pairplot gerado e salvo em outputs/pairplot.png")

        caixa_dica("""
        Observe no pairplot como as três espécies formam <b>agrupamentos visíveis</b>
        em vários pares de variáveis — especialmente comprimento_bico_mm × profundidade_bico_mm.
        Isso indica que um classificador baseado nessas features teria boa performance.
        """)

    divider()
    with st.expander("📝 Ver código desta seção"):
        code_block("""
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = sns.load_dataset("penguins").dropna().reset_index(drop=True)
num_cols = df.select_dtypes(include="float64").columns.tolist()

# --- Histograma por categoria ---
fig, ax = plt.subplots(figsize=(9, 4))
sns.histplot(data=df, x="comprimento_bico_mm", hue="espécie",
             kde=True, element="step", palette="husl", ax=ax)
ax.set_title("Distribuição de comprimento_bico_mm por espécie")
plt.tight_layout()
plt.show()

# --- Boxplot por categoria ---
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x="espécie", y="comprimento_nadadeira_mm",
            palette="husl", width=0.5, ax=ax)
ax.set_title("comprimento_nadadeira_mm por espécie")
plt.tight_layout()
plt.show()

# --- Mapa de correlação ---
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, ax=ax)
ax.set_title("Matriz de Correlação")
plt.tight_layout()
plt.show()

# --- Pairplot ---
fig = sns.pairplot(df, hue="espécie", vars=num_cols,
                   diag_kind="kde", palette="husl",
                   plot_kws={"alpha": 0.6})
fig.fig.suptitle("Pairplot — Palmer Penguins", y=1.02)
plt.show()
""")


# ============================================================================
# PÁGINA: TRATAMENTO DE OUTLIERS
# ============================================================================

elif pagina == "🎯  Tratamento de Outliers":

    st.markdown("""
    <div style='text-align:center; padding: 1.5rem 0 0.5rem;'>
        <h1 style='font-size:2.4rem; font-weight:800; color:#1a1a2e;'>
            🎯 Tratamento de Outliers
        </h1>
        <p style='font-size:1.05rem; color:#555;'>
            Valores extremos que distorcem modelos — identifique e decida o que fazer.
        </p>
    </div>
    """, unsafe_allow_html=True)

    divider()

    df_clean = dados_limpos()
    df_no_out = dados_sem_outliers()
    num_cols = df_clean.select_dtypes(include="float64").columns.tolist()

    caixa_conceito("""
    <b>O que é um outlier?</b><br>
    Um outlier é uma observação que se distancia significativamente do padrão dos demais dados.
    Pode ser um <b>erro de medição</b>, um <b>erro de digitação</b> ou um valor
    <b>genuíno, porém extremo</b>. Outliers afetam a média, o desvio padrão e,
    consequentemente, a performance de algoritmos sensíveis a escala como Regressão
    Linear, KNN e SVM.
    """)

    divider()

    # --- Método IQR ---
    secao("📐 O Método IQR (Intervalo Interquartil)")

    caixa_conceito("""
    <b>Como funciona o método IQR?</b><br>
    <ol>
      <li>Calcule o 1º quartil: <code>Q1 = df[col].quantile(0.25)</code></li>
      <li>Calcule o 3º quartil: <code>Q3 = df[col].quantile(0.75)</code></li>
      <li>Calcule o IQR: <code>IQR = Q3 - Q1</code></li>
      <li>Defina os limites:<br>
          &nbsp;&nbsp;&nbsp;&nbsp;<code>limite_inferior = Q1 - 1.5 × IQR</code><br>
          &nbsp;&nbsp;&nbsp;&nbsp;<code>limite_superior = Q3 + 1.5 × IQR</code>
      </li>
      <li>Qualquer valor fora desses limites é considerado outlier.</li>
    </ol>
    Aplicamos o método <b>por espécie</b> para respeitar que cada espécie tem
    morfologia diferente — um outlier da espécie Gentoo não deve ser julgado
    pelos parâmetros da espécie Adelie.
    """)

    divider()

    # --- Métricas antes/depois ---
    secao("📊 Antes × Depois do Tratamento de Outliers")

    col1, col2, col3 = st.columns(3)
    col1.metric("Linhas antes", len(df_clean))
    col2.metric("Linhas após remoção", len(df_no_out))
    col3.metric(
        "Outliers removidos",
        len(df_clean) - len(df_no_out),
        delta=f"-{len(df_clean) - len(df_no_out)}",
        delta_color="inverse",
    )

    divider()

    # --- Visualização interativa antes/depois ---
    secao("🔍 Comparação Interativa: Antes × Depois")

    col_sel = st.selectbox("Selecione a variável para inspecionar outliers:", options=num_cols)

    col_antes, col_depois = st.columns(2)

    with col_antes:
        st.markdown("**ANTES — Dataset limpo (333 linhas)**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.boxplot(
            data=df_clean,
            x="espécie",
            y=col_sel,
            palette="husl",
            width=0.5,
            ax=ax,
        )
        ax.set_title(f"{col_sel}\n(antes)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Espécie")
        ax.set_ylabel(col_sel)
        fig.tight_layout()
        save_and_show(fig, f"outlier_antes_{col_sel}")

    with col_depois:
        st.markdown("**DEPOIS — Sem outliers IQR**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.boxplot(
            data=df_no_out,
            x="espécie",
            y=col_sel,
            palette="husl",
            width=0.5,
            ax=ax,
        )
        ax.set_title(f"{col_sel}\n(depois)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Espécie")
        ax.set_ylabel(col_sel)
        fig.tight_layout()
        save_and_show(fig, f"outlier_depois_{col_sel}")

    divider()

    # --- Detalhes por coluna ---
    secao("🔢 Limites IQR calculados por espécie")

    with st.expander("Ver tabela de limites por espécie e coluna"):
        rows = []
        for species in df_clean["espécie"].unique():
            grp = df_clean[df_clean["espécie"] == species]
            for col in num_cols:
                q1 = grp[col].quantile(0.25)
                q3 = grp[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                n_out = ((grp[col] < lower) | (grp[col] > upper)).sum()
                rows.append({
                    "Espécie": species,
                    "Coluna": col,
                    "Q1": round(q1, 2),
                    "Q3": round(q3, 2),
                    "IQR": round(iqr, 2),
                    "Limite Inf.": round(lower, 2),
                    "Limite Sup.": round(upper, 2),
                    "Outliers": n_out,
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    divider()

    caixa_atencao("""
    <b>Cuidado ao remover outliers:</b>
    Nem todo valor extremo é um erro. Um pinguim genuinamente grande ou pequeno
    é uma observação legítima. Antes de remover, sempre pergunte:
    <ol>
      <li>É um erro de digitação ou medição?</li>
      <li>Faz sentido biologicamente?</li>
      <li>Quantos dados perderei? Vale a pena?</li>
    </ol>
    Neste exercício didático removemos outliers para praticar o método —
    em produção, a decisão deve ser sempre documentada e justificada.
    """)

    divider()

    # --- Download ---
    secao("💾 Dataset final limpo")

    st.markdown(f"O dataset final possui **{len(df_no_out)} linhas** e **{df_no_out.shape[1]} colunas**.")
    st.dataframe(df_no_out.head(10), use_container_width=True)

    csv_bytes = df_no_out.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Baixar dataset limpo (CSV)",
        data=csv_bytes,
        file_name="penguins_clean_no_outliers.csv",
        mime="text/csv",
    )

    divider()

    with st.expander("📝 Ver código desta seção"):
        code_block("""
import seaborn as sns
import pandas as pd

# Dataset limpo (sem nulos e sem duplicatas)
df = sns.load_dataset("penguins").dropna().drop_duplicates().reset_index(drop=True)
print(f"Antes: {len(df)} linhas")  # 333

num_cols = df.select_dtypes(include="float64").columns.tolist()

# ---- Função para criar máscara IQR dentro de um grupo ----
def iqr_mask(grupo: pd.DataFrame, col: str) -> pd.Series:
    Q1 = grupo[col].quantile(0.25)
    Q3 = grupo[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return (grupo[col] >= lower) & (grupo[col] <= upper)

# ---- Aplicar o filtro IQR por espécie em todas as colunas numéricas ----
mask = pd.Series(True, index=df.index)

for col in num_cols:
    # groupby preserva o índice original com group_keys=False
    col_mask = df.groupby("espécie", group_keys=False).apply(
        lambda g: iqr_mask(g, col)
    )
    mask = mask & col_mask

df_clean = df[mask].reset_index(drop=True)
print(f"Depois: {len(df_clean)} linhas")   # ~327

removidos = len(df) - len(df_clean)
print(f"Outliers removidos: {removidos}")

# Salvar o resultado
df_clean.to_csv("penguins_clean_no_outliers.csv", index=False)

# Visualizar antes e depois
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
import seaborn as sns

sns.boxplot(data=df, x="espécie", y="comprimento_bico_mm",
            palette="husl", ax=axes[0])
axes[0].set_title("ANTES — comprimento_bico_mm")

sns.boxplot(data=df_clean, x="espécie", y="comprimento_bico_mm",
            palette="husl", ax=axes[1])
axes[1].set_title("DEPOIS — comprimento_bico_mm")

plt.tight_layout()
plt.show()
""")

    # --- Footer ---
    st.markdown("""
    <div class="footer">
        <p style="margin:0; font-size:1rem; font-weight:600;">
            Machine Learning — Aula 02 · Análise Exploratória de Dados
        </p>
        <p style="margin:0.3rem 0 0; font-size:0.85rem; opacity:0.7;">
            Dataset limpo salvo em <code>outputs/</code> e disponível para download acima.
        </p>
    </div>
    """, unsafe_allow_html=True)
