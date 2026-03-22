# Padrão Unificado — Machine Learning

> **Autor:** Cláudio Ferreira Neves · Especialista em Ciência de Dados e IA
> Este documento define o padrão único que deve ser seguido em **todos** os artefatos do projeto: notebooks, Streamlit pages e READMEs.

---

## 1. Estrutura de Pastas

```
machine_learning/
├── app.py                          ← Portal de navegação (não alterar)
├── nav.py                          ← Componente de abas (não alterar)
├── README.md                       ← README raiz do projeto
├── pages/
│   └── Aula_NN.py                  ← Thin wrapper — não alterar (gerado automaticamente)
│
└── aula_NN/                        ← Uma pasta por aula
    ├── Aula_NN_Tema_(resolvido).ipynb   ← Notebook completo (professor)
    ├── Aula_NN_Tema_(aluno).ipynb       ← Notebook com exercícios em branco (aluno)
    ├── Slides_Aula_NN_Tema.pdf          ← Material teórico
    ├── app_streamlit.py                 ← Aplicação Streamlit interativa
    ├── aulaNNtema.py                    ← Script Python puro (sem interface)
    ├── requirements.txt                 ← Dependências da aula
    ├── README.md                        ← Documentação da aula
    └── outputs/                         ← Gráficos gerados (criada automaticamente)
```

---

## 2. Convenção de Nomes

| Artefato | Padrão | Exemplo |
|----------|--------|---------|
| Pasta da aula | `aula_NN/` (dois dígitos) | `aula_03/` |
| Notebook professor | `Aula_NN_Tema_(resolvido).ipynb` | `Aula_03_Regressao_Linear_(resolvido).ipynb` |
| Notebook aluno | `Aula_NN_Tema_(aluno).ipynb` | `Aula_03_Regressao_Linear_(aluno).ipynb` |
| Slides PDF | `Slides_Aula_NN_Tema.pdf` | `Slides_Aula_03_Regressao_Linear.pdf` |
| Script Python | `aula NN_tema.py` (snake_case) | `aula03_regressao_linear.py` |
| Página Streamlit | `app_streamlit.py` (fixo) | `app_streamlit.py` |
| Outputs | `NN_descricao.png` (prefixo numérico) | `01_scatter.png`, `02_reta.png` |

---

## 3. Padrão de README

Ver arquivo `README_TEMPLATE.md` nesta pasta.

**Seções obrigatórias (nesta ordem):**

1. Cabeçalho com emoji + número + tema
2. Autor + aviso de plágio
3. "Sobre esta aula" — contexto e motivação
4. "Como o projeto está organizado" — árvore de arquivos
5. "Como rodar na sua máquina" — pip install + comandos
6. "O que você vai aprender" — tabela de seções
7. "O que dá pra mexer em tempo real" — controles Streamlit
8. "Os conceitos principais" — teoria + fórmulas
9. "As métricas de avaliação" — tabela (quando aplicável)
10. "Os datasets utilizados" — tabela (quando aplicável)
11. "As ferramentas do projeto" — tabela resumida
12. "Os gráficos são salvos automaticamente" — parágrafo padrão
13. "Autor" — bloco final

---

## 4. Padrão de Notebook

Ver arquivo `notebook_template.ipynb` nesta pasta.

**Estrutura de células (ordem fixa):**

```
[CÉLULA 0]  Markdown — Cabeçalho (curso, aula, título, autor)
[CÉLULA 1]  Markdown — Objetivos de aprendizagem + pré-requisitos
[CÉLULA 2]  Code     — Importações globais comentadas

──── Seção N ────────────────────────────────────────────────
[CÉLULA]    Markdown — ## Seção N — Título
[CÉLULA]    Markdown — Conceito + analogia + fórmula (se houver)
[CÉLULA]    Code     — Código comentado linha a linha
[CÉLULA]    Markdown — > 🎯 Momento de Praticar
[CÉLULA]    Code     — Célula vazia (na versão aluno) / resolvida (na versão professor)
──────────────────────────────────────────────────────────────

[ÚLTIMA]    Markdown — Resumo: tabela conceito | definição | quando usar
                       Checklist do que foi aprendido
                       Próximos passos
```

**Regras obrigatórias:**
- Toda célula de código deve ter ao menos um comentário explicativo
- Fórmulas matemáticas em LaTeX inline: `$formula$` ou bloco `$$formula$$`
- Células de exercício marcadas com `🎯` no título da seção de prática
- Variáveis com nomes descritivos em português ou inglês padrão da área
- `np.random.seed(42)` sempre que houver aleatoriedade (reprodutibilidade)

---

## 5. Padrão de Streamlit Page (app_streamlit.py)

Ver arquivo `app_streamlit_template.py` nesta pasta.

**Estrutura obrigatória:**

```python
# 1. Docstring do módulo (autor, aula, cargo)
# 2. Imports (os, sys, numpy, matplotlib, streamlit, sklearn...)
# 3. matplotlib.use("Agg")  ← OBRIGATÓRIO antes de qualquer plt
# 4. Constantes: PAGE_PORTAL, PAGE_PROXIMA_AULA, OUTPUTS_DIR
# 5. os.makedirs(OUTPUTS_DIR, exist_ok=True)
# 6. Funções auxiliares: save_and_show(), plot_decision_boundary()
# 7. Uma função render_NOME() por seção da aula
# 8. Função main() com: st.set_page_config → st.title → navegação → tabs/seções
# 9. if __name__ == "__main__": main()
```

**Regras obrigatórias:**
- `st.set_page_config(page_title=..., page_icon=..., layout="wide")` sempre no topo de `main()`
- Cada seção encapsulada em sua própria função `render_NOME()`
- Sliders e controles sempre com `key=` único para evitar conflitos
- `st.info()` para conceitos teóricos, `st.warning()` para alertas, `st.success()` para resultados
- Gráficos sempre salvos via `save_and_show()` — nunca `st.pyplot(fig)` direto
- Navegação: botões `← Aula anterior` e `Próxima aula →` no topo e no rodapé

---

## 6. Emojis por Tema (padronizados)

| Tema | Emoji |
|------|-------|
| Introdução ao ML | 🤖 |
| Análise de Dados / EDA | 🔍 |
| Regressão Linear/Polinomial | 📈 |
| Classificação / Log. Regression / KNN | 🔮 |
| Naive Bayes / SVM | ⚡ |
| Decision Tree / Random Forest | 🌳 |
| K-means / Clusterização | 🎯 |
| PCA / Redução de dimensionalidade | 🔭 |
| Redes Neurais | 🧠 |
| Deploy / Pipelines | 🚀 |

---

## 7. Checklist antes de publicar uma aula

- [ ] Notebook `(resolvido)` executa do início ao fim sem erros
- [ ] Notebook `(aluno)` tem células de exercício vazias (marcadas com 🎯)
- [ ] `app_streamlit.py` roda com `streamlit run` sem erros
- [ ] README tem todas as seções obrigatórias
- [ ] Nomes de arquivos seguem a convenção de nomes
- [ ] `requirements.txt` lista todas as dependências da aula
- [ ] Pasta `outputs/` criada automaticamente pelo script (não commitada vazia)
- [ ] `np.random.seed(42)` presente em todo código com aleatoriedade
