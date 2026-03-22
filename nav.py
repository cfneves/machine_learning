"""
Navegação entre aulas — componente compartilhado.
Renderiza uma barra de abas centralizada no topo de cada aula.
"""
import streamlit as st

_ITENS = [
    ("pages/Portal.py",  "🏠",       "Portal"),
    ("pages/Aula_01.py", "Aula 01",  "Introdução ao ML"),
    ("pages/Aula_02.py", "Aula 02",  "EDA"),
    ("pages/Aula_03.py", "Aula 03",  "Regressão Linear"),
    ("pages/Aula_04.py", "Aula 04",  "Log. + KNN"),
    ("pages/Aula_05.py", "Aula 05",  "NB + SVM"),
    ("pages/Aula_06.py", "Aula 06",  "Árvores"),
    ("pages/Aula_07.py", "Aula 07",  "K-Means + PCA"),
    ("pages/Aula_08.py", "Aula 08",  "Comparação"),
]


def tab_nav(current: int) -> None:
    """
    Renderiza a barra de abas centralizada.

    Parameters
    ----------
    current : int
        Número da aula atual (1 a 8). Use 0 para o Portal.
    """
    st.markdown("""
<style>
/* Oculta a lista de páginas gerada automaticamente pelo Streamlit na sidebar */
[data-testid="stSidebarNav"] { display: none !important; }

/* Estilo base dos links de página (abas inativas) */
[data-testid="stPageLink"] {
    padding: 0 !important;
    margin: 0 !important;
}
[data-testid="stPageLink"] a {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 0.45rem 0.3rem !important;
    border-radius: 10px !important;
    background: #f0f4ff !important;
    border: 1px solid #c7d2fe !important;
    color: #3730a3 !important;
    font-size: 0.62rem !important;
    line-height: 1.35 !important;
    text-decoration: none !important;
    text-align: center !important;
    width: 100% !important;
    min-height: 54px !important;
    box-sizing: border-box !important;
    transition: background 0.18s, color 0.18s, box-shadow 0.18s !important;
}
[data-testid="stPageLink"] a:hover {
    background: #667eea !important;
    color: white !important;
    border-color: #667eea !important;
    box-shadow: 0 3px 10px rgba(102,126,234,0.35) !important;
}
[data-testid="stPageLink"] a div,
[data-testid="stPageLink"] a p,
[data-testid="stPageLink"] a span,
[data-testid="stPageLink"] a * {
    margin: 0 !important;
    padding: 0 !important;
    color: inherit !important;
    font-size: 0.62rem !important;
    line-height: 1.35 !important;
    text-align: center !important;
}

/* Aba ativa (div estática, sem clique) */
.tab-active {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 0.45rem 0.3rem;
    border-radius: 10px;
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #667eea;
    color: white;
    font-size: 0.62rem;
    line-height: 1.35;
    text-align: center;
    width: 100%;
    min-height: 54px;
    cursor: default;
    box-shadow: 0 3px 10px rgba(102,126,234,0.4);
    box-sizing: border-box;
}
.tab-active b { display: block; font-size: 0.62rem; margin-bottom: 2px; }
.tab-active span { opacity: 0.88; font-weight: 400; font-size: 0.62rem; }
</style>
""", unsafe_allow_html=True)

    cols = st.columns(len(_ITENS))
    for i, (col, (page, num, titulo)) in enumerate(zip(cols, _ITENS)):
        is_active = i == current
        with col:
            if is_active:
                st.markdown(
                    f'<div class="tab-active"><b>{num}</b>'
                    f'<span>{titulo}</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.page_link(
                    page,
                    label=f"{num}\n\n{titulo}",
                    use_container_width=True,
                )
