# Aula 02 — Análise Exploratória de Dados (EDA)

<div align="center">

**Cláudio Ferreira Neves  ·  Analista de BI — Save Co. | Docente SENAI/SC**

</div>

> Material didático autoral. Reprodução ou uso sem crédito ao autor é considerado plágio.

---

## O que é esse projeto?

Antes de treinar qualquer modelo, é preciso entender os dados. Esta aula percorre a **Análise Exploratória de Dados (EDA)** passo a passo: a fase em que o cientista de dados examina os dados antes de qualquer modelagem.

O projeto usa o **dataset Palmer Penguins**, uma alternativa ao clássico Iris com mais variabilidade e contexto biológico, para demonstrar técnicas reais de EDA: inspeção da estrutura, limpeza, análise univariada, multivariada e tratamento de outliers. Tudo alinhado com o **CRISP-DM**, metodologia amplamente usada em projetos de Ciência de Dados.

---

## Como o projeto está organizado

```
aula_02/
│
├── aula02_analise_de_dados.py    # Script Python principal — roda no terminal
├── README.md                     # Este arquivo — documentação do projeto
│
└── outputs/                      # Gráficos salvos automaticamente (150 DPI)
    ├── 01_categoricas_barras.png
    ├── 02_year_barras.png
    ├── 03_numericas_histogramas.png
    ├── 04_numericas_boxplots.png
    ├── 05_multivariada_histogramas_especie.png
    ├── 06_multivariada_boxplots_especie.png
    ├── 07_multivariada_correlacao_heatmap.png
    ├── 08_multivariada_pairplot_especie.png
    └── 09_outliers_antes_depois_boxplot.png
```

> A pasta `outputs/` é criada automaticamente na primeira execução — não é necessário criá-la manualmente.

---

## Como rodar na sua máquina

### 1. Instale as dependências

```bash
pip install numpy pandas matplotlib seaborn
```

### 2. Execute o script

```bash
python aula02_analise_de_dados.py
```

Os resultados aparecem no terminal e os gráficos são salvos automaticamente na pasta `outputs/`. Nenhuma janela gráfica é aberta (o backend `Agg` do Matplotlib é usado para garantir compatibilidade em qualquer ambiente).

---

## Metodologia CRISP-DM

O CRISP-DM (Cross-Industry Standard Process for Data Mining) organiza projetos de Ciência de Dados em **6 fases cíclicas**:

| # | Fase | Descrição |
|---|------|-----------|
| 1 | Entendimento do Negócio | Definição do problema e critérios de sucesso |
| 2 | Entendimento dos Dados | Exploração e avaliação da qualidade dos dados |
| 3 | Preparação dos Dados | Limpeza, transformações e engenharia de features |
| 4 | Modelagem | Seleção e treinamento de algoritmos |
| 5 | Avaliação | Verificação do desempenho e adequação ao negócio |
| 6 | Implantação | Deploy e monitoramento do modelo em produção |

Nesta aula, as fases **2 (Entendimento dos Dados)** e **3 (Preparação dos Dados)** são percorridas em profundidade com o dataset Palmer Penguins.

---

## O que você vai aprender — 6 seções

| # | Seção | O que é trabalhado |
|---|-------|--------------------|
| 1 | **Aquisição dos Dados** | Carregamento do dataset via `sns.load_dataset()`, inspeção com `head()` e `tail()` |
| 2 | **Estrutura dos Dados** | `shape`, `info()`, `dtypes`, `columns`, `nunique()`, loop de valores únicos, contagem de nulos e duplicatas |
| 3 | **Limpeza de Dados** | `dropna()`, verificação pós-limpeza, reset de índice, identificação de colunas numéricas e categóricas |
| 4 | **Análise Univariada** | `describe()`, `value_counts()`, gráficos de barras, histogramas (regra da raiz quadrada), boxplots |
| 5 | **Análise Multivariada** | Histogramas/boxplots por espécie, matriz de correlação (heatmap), pairplot colorido por espécie |
| 6 | **Tratamento de Outliers** | Método IQR aplicado por espécie via função `remove_outliers_iqr()`, comparação visual antes × depois |

---

## O dataset — Palmer Penguins

O **Palmer Penguins** foi coletado na Antártida entre 2007 e 2009 pela pesquisadora Dr. Kristen Gorman em parceria com a Estação Palmer. Tem mais variabilidade que o Iris clássico e um contexto biológico mais rico.

### Variáveis

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `species` | Categórica | Espécie: Adelie, Chinstrap ou Gentoo |
| `island` | Categórica | Ilha de origem: Torgersen, Biscoe ou Dream |
| `bill_length_mm` | Numérica | Comprimento do bico (mm) |
| `bill_depth_mm` | Numérica | Profundidade do bico (mm) |
| `flipper_length_mm` | Numérica | Comprimento da nadadeira (mm) |
| `body_mass_g` | Numérica | Massa corporal (g) |
| `sex` | Categórica | Sexo: Male ou Female |
| `year` | Numérica | Ano de coleta: 2007, 2008 ou 2009 |

### Distribuição por espécie (dataset original)

| Espécie | Registros |
|---------|-----------|
| Adelie | 152 |
| Gentoo | 124 |
| Chinstrap | 68 |
| **Total** | **344** |

> Após `dropna()`, o dataset fica com **333 linhas** (11 com valores nulos removidos).

---

## As ferramentas do projeto

### pandas — manipulação e análise de dados

A biblioteca central para trabalhar com dados tabulares em Python. Oferece o `DataFrame` (tabela de dados) e o `Series` (coluna), com operações de filtragem, agrupamento, limpeza e transformação.

**O que ela faz neste projeto:**
- Carrega e armazena o dataset como DataFrame
- Realiza a limpeza com `dropna()` e `duplicated()`
- Calcula estatísticas com `describe()`, `value_counts()` e `groupby()`
- Filtra e indexa dados para a remoção de outliers

```python
import pandas as pd

df.isnull().sum()           # conta nulos por coluna
df.dropna(inplace=True)     # remove linhas com qualquer nulo
df.describe()               # estatísticas descritivas completas
df['species'].value_counts() # frequência de cada categoria
```

---

### NumPy — operações numéricas e arrays

A base numérica do ecossistema Python científico. Fornece arrays de alta performance e funções matemáticas vetorizadas.

**O que ela faz neste projeto:**
- Calcula o número de bins para histogramas pela regra da raiz quadrada
- Cria a máscara triangular para o heatmap de correlação

```python
import numpy as np

n_bins = int(np.sqrt(len(df)))   # regra da raiz quadrada para bins
mask = np.triu(np.ones_like(corr, dtype=bool))  # triângulo superior
```

---

### Matplotlib — visualizações base

A biblioteca mais tradicional para criação de gráficos em Python. Oferece controle total sobre cada elemento visual.

**O que ela faz neste projeto:**
- Cria histogramas, boxplots e gráficos de barras
- Salva todas as figuras em PNG com 150 DPI via `savefig()`
- Opera em modo não-interativo com o backend `Agg`

```python
import matplotlib
matplotlib.use('Agg')           # sem janela gráfica — compatível com qualquer ambiente
import matplotlib.pyplot as plt

plt.savefig('outputs/fig.png', dpi=150)
plt.close()
```

---

### Seaborn — visualizações estatísticas

Construído sobre o Matplotlib, o Seaborn é especializado em gráficos estatísticos. Integra-se nativamente com DataFrames do pandas e produz visualizações elegantes com poucas linhas de código.

**O que ela faz neste projeto:**
- Carrega o dataset Palmer Penguins via `sns.load_dataset('penguins')`
- Cria boxplots agrupados por espécie com `sns.boxplot()`
- Gera o heatmap da matriz de correlação com `sns.heatmap()`
- Produz o pairplot com `sns.pairplot()`

```python
import seaborn as sns

df = sns.load_dataset('penguins')   # carrega o dataset integrado

sns.boxplot(data=df, x='species', y='bill_length_mm', palette='Set2')

sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)

sns.pairplot(df, hue='species', diag_kind='hist', corner=True)
```

---

## As fórmulas — entenda a matemática por trás

### Estatísticas Descritivas Básicas

A média (tendência central) e o desvio padrão (dispersão) são os primeiros indicadores calculados em qualquer EDA:

$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i \qquad s = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2}$$

### Número de Bins — Regra da Raiz Quadrada

Para definir quantas barras usar em um histograma sem distorcer a distribuição:

$$k = \lfloor\sqrt{n}\rfloor$$

Onde $n$ é o número de observações e $\lfloor\cdot\rfloor$ é o piso (arredondamento para baixo).

### Correlação de Pearson

Mede a força e direção da relação linear entre duas variáveis numéricas. Varia entre -1 (correlação negativa perfeita) e +1 (correlação positiva perfeita):

$$r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2 \cdot \sum_{i=1}^{n}(y_i - \bar{y})^2}}$$

### Remoção de Outliers — Método IQR

O **IQR (Interquartile Range — Intervalo Interquartil)** é a diferença entre o 3º e o 1º quartil. Valores além dos limites calculados abaixo são classificados como outliers e removidos:

$$\text{IQR} = Q_3 - Q_1$$

$$\text{Limite inferior} = Q_1 - 1{,}5 \times \text{IQR}$$

$$\text{Limite superior} = Q_3 + 1{,}5 \times \text{IQR}$$

No script, essa remoção é realizada **dentro de cada espécie** via a função `remove_outliers_iqr(df, column, category_col='species')`, preservando a variação natural entre as populações de pinguins.

---

## Os gráficos são salvos automaticamente

Toda vez que o script é executado, os gráficos são exportados para a pasta `outputs/`:

- Formato: **PNG** (Portable Network Graphics)
- Resolução: **150 DPI** (qualidade adequada para impressão e apresentações)
- Nomeados com prefixo numérico sequencial para facilitar a identificação
- A pasta é criada automaticamente via `os.makedirs(exist_ok=True)`

---

## Autor

**Cláudio Ferreira Neves**
Analista de BI — Save Co. | Docente SENAI/SC

> Este material é de autoria exclusiva de **Cláudio Ferreira Neves**.
> É permitido o uso para fins de estudo, desde que citada a fonte.
> Reprodução total ou parcial sem crédito ao autor caracteriza **plágio**.
