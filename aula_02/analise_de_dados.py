"""
Machine Learning
Aula 02 - Análise Exploratória de Dados (EDA)

Autor: Cláudio Ferreira Neves
Cargo: Analista de BI — Save Co. | Docente SENAI/SC
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo: salva figuras sem abrir janela
import matplotlib.pyplot as plt
import seaborn as sns

# Pasta de saída para os gráficos
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Contador global de figuras salvas
_fig_counter = [0]

def save_fig(name):
    """Salva a figura atual em outputs/ com numeração sequencial e 150 DPI."""
    _fig_counter[0] += 1
    filename = f"{_fig_counter[0]:02d}_{name}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  [Figura salva] {filename}")


# ==============================================================================
# METODOLOGIA CRISP-DM
# ==============================================================================

print("=" * 60)
print("METODOLOGIA CRISP-DM")
print("=" * 60)

print("""
CRISP-DM (Cross-Industry Standard Process for Data Mining)
é o processo mais utilizado em projetos de Ciência de Dados.
Ele organiza o trabalho em 6 fases cíclicas:

  1. Entendimento do Negócio (Business Understanding)
     - Qual problema queremos resolver?
     - Qual o critério de sucesso?

  2. Entendimento dos Dados (Data Understanding)
     - Quais dados temos disponíveis?
     - Os dados são suficientes e confiáveis?

  3. Preparação dos Dados (Data Preparation)
     - Limpeza, tratamento de nulos e outliers
     - Transformações e engenharia de features

  4. Modelagem (Modeling)
     - Escolha e treinamento dos algoritmos de ML
     - Ajuste de hiperparâmetros

  5. Avaliação (Evaluation)
     - O modelo atende aos critérios do negócio?
     - Métricas de desempenho (R², Accuracy, F1 etc.)

  6. Implantação (Deployment)
     - Colocar o modelo em produção
     - Monitorar e manter o modelo ao longo do tempo

Nesta aula, percorreremos as fases 2 e 3 com o
dataset Palmer Penguins, realizando uma EDA completa.
""")


# ==============================================================================
# SEÇÃO 1 — AQUISIÇÃO DOS DADOS
# ==============================================================================

print("=" * 60)
print("SEÇÃO 1 — AQUISIÇÃO DOS DADOS")
print("=" * 60)

# O dataset Palmer Penguins foi coletado na Antártida (2007-2009) pela
# pesquisadora Dr. Kristen Gorman e contém medidas morfológicas de
# 344 pinguins de 3 espécies: Adelie, Chinstrap e Gentoo.
df = sns.load_dataset('penguins')

print(f"\nDataset carregado com sucesso via sns.load_dataset('penguins')")
print(f"Tipo retornado: {type(df)}")
print(f"\nColunas disponíveis:")
for col in df.columns:
    print(f"  - {col}")

print(f"\nPrimeiras 5 linhas (df.head()):")
print(df.head())

print(f"\nÚltimas 5 linhas (df.tail()):")
print(df.tail())


# ==============================================================================
# SEÇÃO 2 — ESTRUTURA DOS DADOS
# ==============================================================================

print("\n" + "=" * 60)
print("SEÇÃO 2 — ESTRUTURA DOS DADOS")
print("=" * 60)

# --- Dimensões e tipos ---
print(f"\nDimensões (df.shape): {df.shape}")
print(f"  → {df.shape[0]} linhas  |  {df.shape[1]} colunas")

print(f"\nInformações gerais (df.info()):")
print(df.info())

print(f"\nTipos de dados (df.dtypes):")
print(df.dtypes)

print(f"\nNomes das colunas (df.columns):")
print(list(df.columns))

print(f"\nQuantidade de valores únicos por coluna (df.nunique()):")
print(df.nunique())

# Valores únicos por coluna (detalhe)
print(f"\nValores únicos por coluna:")
for col in df.columns:
    unicos = df[col].dropna().unique()
    if len(unicos) <= 10:
        print(f"  {col}: {list(unicos)}")
    else:
        print(f"  {col}: {len(unicos)} valores únicos "
              f"(min={df[col].min():.2f}, max={df[col].max():.2f})")

# --- Valores nulos ---
print(f"\nValores nulos por coluna (df.isnull().sum()):")
nulos = df.isnull().sum()
pct_nulos = (nulos / len(df) * 100).round(2)
resumo_nulos = pd.DataFrame({'nulos': nulos, 'percentual (%)': pct_nulos})
print(resumo_nulos)

# --- Duplicatas ---
n_dup = df.duplicated().sum()
print(f"\nLinhas duplicadas (df.duplicated().sum()): {n_dup}")


# ==============================================================================
# SEÇÃO 3 — LIMPEZA DE DADOS
# ==============================================================================

print("\n" + "=" * 60)
print("SEÇÃO 3 — LIMPEZA DE DADOS")
print("=" * 60)

print(f"\nLinhas antes da limpeza: {len(df)}")

# Remover linhas com valores nulos
df.dropna(inplace=True)
print(f"Linhas após dropna():    {len(df)}")
print(f"Linhas removidas:        {344 - len(df)}")

# Verificar nulos novamente
print(f"\nValores nulos após limpeza:")
print(df.isnull().sum())

# Verificar duplicatas após limpeza
n_dup_pos = df.duplicated().sum()
print(f"\nDuplicatas após limpeza: {n_dup_pos}")

# Resetar índice
df = df.reset_index(drop=True)
print(f"\nDataset final: {df.shape[0]} linhas × {df.shape[1]} colunas")

# Identificar colunas numéricas e categóricas
colunas_numericas = df.select_dtypes(include='number').columns.tolist()
colunas_categoricas = df.select_dtypes(include='object').columns.tolist()

print(f"\nColunas numéricas:   {colunas_numericas}")
print(f"Colunas categóricas: {colunas_categoricas}")


# ==============================================================================
# SEÇÃO 4 — ANÁLISE UNIVARIADA
# ==============================================================================

print("\n" + "=" * 60)
print("SEÇÃO 4 — ANÁLISE UNIVARIADA")
print("=" * 60)

# --- Estatísticas descritivas ---
print(f"\nEstatísticas descritivas (df.describe()):")
print(df.describe().round(2))

print(f"\nEstatísticas descritivas — colunas categóricas (df.describe(include='object')):")
print(df.describe(include='object'))

# ---- Gráficos de barras — variáveis categóricas ----
print(f"\nFrequências — variáveis categóricas:")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Distribuição das Variáveis Categóricas", fontsize=14, fontweight='bold')

cat_cols = ['species', 'island', 'sex']
cores = ['#4C72B0', '#55A868', '#C44E52']

for ax, col, cor in zip(axes, cat_cols, cores):
    contagem = df[col].value_counts()
    print(f"\n  {col}:")
    print(contagem.to_string())
    contagem.plot(kind='bar', ax=ax, color=cor, edgecolor='black', alpha=0.85)
    ax.set_title(f"Distribuição: {col}", fontsize=11)
    ax.set_xlabel(col.capitalize())
    ax.set_ylabel("Contagem")
    ax.tick_params(axis='x', rotation=30)
    for p in ax.patches:
        ax.annotate(str(int(p.get_height())),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10)

save_fig("categoricas_barras")

# Gráfico de barras — year
fig, ax = plt.subplots(figsize=(7, 5))
contagem_year = df['year'].value_counts().sort_index()
print(f"\n  year:")
print(contagem_year.to_string())
contagem_year.plot(kind='bar', ax=ax, color='#8172B2', edgecolor='black', alpha=0.85)
ax.set_title("Distribuição por Ano de Coleta", fontsize=12, fontweight='bold')
ax.set_xlabel("Ano")
ax.set_ylabel("Contagem")
ax.tick_params(axis='x', rotation=0)
for p in ax.patches:
    ax.annotate(str(int(p.get_height())),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=11)
save_fig("year_barras")

# ---- Histogramas — variáveis numéricas ----
print(f"\nHistogramas — variáveis numéricas:")

# Número de bins pela regra da raiz quadrada
n_bins = int(np.sqrt(len(df)))
print(f"  Número de bins (sqrt rule): int(sqrt({len(df)})) = {n_bins}")

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle(f"Histogramas das Variáveis Numéricas (bins={n_bins})",
             fontsize=14, fontweight='bold')

num_labels = {
    'bill_length_mm':   'Comprimento do bico (mm)',
    'bill_depth_mm':    'Profundidade do bico (mm)',
    'flipper_length_mm': 'Comprimento da nadadeira (mm)',
    'body_mass_g':      'Massa corporal (g)',
}
cores_hist = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']

for ax, (col, label), cor in zip(axes.flat, num_labels.items(), cores_hist):
    ax.hist(df[col], bins=n_bins, color=cor, edgecolor='black', alpha=0.8)
    ax.set_title(label, fontsize=10)
    ax.set_xlabel(label)
    ax.set_ylabel("Frequência")
    media = df[col].mean()
    ax.axvline(media, color='red', linestyle='--', linewidth=1.5,
               label=f'Média: {media:.1f}')
    ax.legend(fontsize=9)

save_fig("numericas_histogramas")

# ---- Boxplots — variáveis numéricas ----
print(f"\nBoxplots — variáveis numéricas:")

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Boxplots das Variáveis Numéricas", fontsize=14, fontweight='bold')

for ax, (col, label), cor in zip(axes.flat, num_labels.items(), cores_hist):
    ax.boxplot(df[col], patch_artist=True,
               boxprops=dict(facecolor=cor, alpha=0.7),
               medianprops=dict(color='black', linewidth=2))
    ax.set_title(label, fontsize=10)
    ax.set_ylabel(label)
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    mediana = df[col].median()
    print(f"  {col}: Q1={q1:.2f}  Mediana={mediana:.2f}  Q3={q3:.2f}  "
          f"IQR={q3 - q1:.2f}")

save_fig("numericas_boxplots")


# ==============================================================================
# SEÇÃO 5 — ANÁLISE MULTIVARIADA
# ==============================================================================

print("\n" + "=" * 60)
print("SEÇÃO 5 — ANÁLISE MULTIVARIADA")
print("=" * 60)

# Paleta por espécie (consistente em todos os gráficos)
palette_especies = {
    'Adelie':    '#4C72B0',
    'Chinstrap': '#C44E52',
    'Gentoo':    '#55A868',
}

# ---- Histogramas por espécie ----
print(f"\nHistogramas por espécie (Adelie | Chinstrap | Gentoo):")

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Histogramas por Espécie", fontsize=14, fontweight='bold')

for ax, (col, label) in zip(axes.flat, num_labels.items()):
    for especie, cor in palette_especies.items():
        dados = df[df['species'] == especie][col]
        ax.hist(dados, bins=n_bins, alpha=0.55, color=cor,
                edgecolor='black', linewidth=0.5, label=especie)
    ax.set_title(label, fontsize=10)
    ax.set_xlabel(label)
    ax.set_ylabel("Frequência")
    ax.legend(fontsize=8)

save_fig("multivariada_histogramas_especie")

# ---- Boxplots por espécie ----
print(f"\nBoxplots por espécie:")

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Boxplots por Espécie", fontsize=14, fontweight='bold')

for ax, (col, label) in zip(axes.flat, num_labels.items()):
    sns.boxplot(data=df, x='species', y=col, ax=ax,
                palette=palette_especies, linewidth=1.2)
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("Espécie")
    ax.set_ylabel(label)
    # Estatísticas por espécie
    print(f"\n  {col} — média por espécie:")
    print(df.groupby('species')[col].mean().round(2).to_string())

save_fig("multivariada_boxplots_especie")

# ---- Matriz de correlação ----
print(f"\nMatriz de correlação (Pearson):")

corr = df[colunas_numericas].corr()
print(corr.round(3))

fig, ax = plt.subplots(figsize=(7, 6))
mask = np.triu(np.ones_like(corr, dtype=bool))  # Oculta triângulo superior
sns.heatmap(
    corr, mask=mask, annot=True, fmt='.2f',
    cmap='coolwarm', center=0, vmin=-1, vmax=1,
    linewidths=0.5, ax=ax,
    annot_kws={'size': 11}
)
ax.set_title("Matriz de Correlação — Variáveis Numéricas",
             fontsize=12, fontweight='bold')
save_fig("multivariada_correlacao_heatmap")

# ---- Pairplot por espécie ----
print(f"\nGerando pairplot por espécie (pode levar alguns segundos)...")

pairplot_fig = sns.pairplot(
    df[colunas_numericas + ['species']],
    hue='species',
    palette=palette_especies,
    diag_kind='hist',
    plot_kws={'alpha': 0.6, 'edgecolor': 'none'},
    diag_kws={'alpha': 0.7, 'bins': n_bins},
    corner=True,
)
pairplot_fig.figure.suptitle(
    "Pairplot — Variáveis Numéricas por Espécie",
    fontsize=13, fontweight='bold', y=1.01
)
_fig_counter[0] += 1
pairplot_filename = f"{_fig_counter[0]:02d}_multivariada_pairplot_especie.png"
pairplot_fig.savefig(
    os.path.join(OUTPUT_DIR, pairplot_filename), dpi=150, bbox_inches='tight'
)
plt.close('all')
print(f"  [Figura salva] {pairplot_filename}")


# ==============================================================================
# SEÇÃO 6 — TRATAMENTO DE OUTLIERS (MÉTODO IQR)
# ==============================================================================

print("\n" + "=" * 60)
print("SEÇÃO 6 — TRATAMENTO DE OUTLIERS (MÉTODO IQR)")
print("=" * 60)

print("""
O método IQR (Interquartile Range — Intervalo Interquartil) define
outliers como valores fora dos limites:

  Limite inferior: Q1 - 1.5 × IQR
  Limite superior: Q3 + 1.5 × IQR

Onde:
  Q1  = 1º quartil (25%)
  Q3  = 3º quartil (75%)
  IQR = Q3 - Q1

A remoção é feita por espécie para preservar a variação
natural entre as diferentes populações de pinguins.
""")


def remove_outliers_iqr(dataframe, column, category_col='species'):
    """
    Remove outliers de uma coluna numérica usando o método IQR,
    aplicado separadamente dentro de cada categoria.

    Parâmetros
    ----------
    dataframe    : pd.DataFrame  — dataset de entrada
    column       : str           — coluna numérica a filtrar
    category_col : str           — coluna categórica para agrupamento

    Retorna
    -------
    pd.DataFrame com as linhas de outliers removidas.
    """
    mascara = pd.Series(True, index=dataframe.index)

    for categoria in dataframe[category_col].unique():
        idx_cat = dataframe[category_col] == categoria
        valores = dataframe.loc[idx_cat, column]

        q1  = valores.quantile(0.25)
        q3  = valores.quantile(0.75)
        iqr = q3 - q1

        lim_inf = q1 - 1.5 * iqr
        lim_sup = q3 + 1.5 * iqr

        outliers_cat = ~valores.between(lim_inf, lim_sup)
        n_out = outliers_cat.sum()

        print(f"  [{column}] {categoria:12s} → "
              f"Q1={q1:.2f}  Q3={q3:.2f}  IQR={iqr:.2f}  "
              f"Limites=[{lim_inf:.2f}, {lim_sup:.2f}]  "
              f"Outliers={n_out}")

        mascara.loc[idx_cat[idx_cat].index] = (
            mascara.loc[idx_cat[idx_cat].index] & ~outliers_cat
        )

    return dataframe[mascara].copy()


print(f"\nDataset antes do tratamento: {len(df)} linhas")
df_sem_outliers = df.copy()

for col in colunas_numericas:
    print(f"\nProcessando coluna: {col}")
    antes = len(df_sem_outliers)
    df_sem_outliers = remove_outliers_iqr(df_sem_outliers, col)
    depois = len(df_sem_outliers)
    removidos = antes - depois
    print(f"  Removidos nesta etapa: {removidos} linha(s)")

print(f"\nDataset após tratamento de outliers: {len(df_sem_outliers)} linhas")
print(f"Total de outliers removidos: {len(df) - len(df_sem_outliers)} linhas")
print(f"Dataset reduzido em: {(len(df) - len(df_sem_outliers)) / len(df) * 100:.1f}%")

# ---- Boxplots antes × depois (outliers) ----
print(f"\nGerando comparação visual: antes vs. depois da remoção de outliers...")

fig, axes = plt.subplots(2, 4, figsize=(16, 9))
fig.suptitle("Boxplots por Espécie — Antes × Depois da Remoção de Outliers (IQR)",
             fontsize=13, fontweight='bold')

for i, (col, label) in enumerate(num_labels.items()):
    ax_antes  = axes[0][i]
    ax_depois = axes[1][i]

    sns.boxplot(data=df, x='species', y=col, ax=ax_antes,
                palette=palette_especies, linewidth=1.2)
    ax_antes.set_title(f"ANTES\n{label}", fontsize=9)
    ax_antes.set_xlabel("")
    ax_antes.set_ylabel(label if i == 0 else "")
    ax_antes.tick_params(axis='x', rotation=20)

    sns.boxplot(data=df_sem_outliers, x='species', y=col, ax=ax_depois,
                palette=palette_especies, linewidth=1.2)
    ax_depois.set_title(f"DEPOIS\n{label}", fontsize=9)
    ax_depois.set_xlabel("Espécie")
    ax_depois.set_ylabel(label if i == 0 else "")
    ax_depois.tick_params(axis='x', rotation=20)

save_fig("outliers_antes_depois_boxplot")

# ---- Resumo estatístico após limpeza ----
print(f"\nEstatísticas descritivas após remoção de outliers:")
print(df_sem_outliers[colunas_numericas].describe().round(2))

print(f"\nDistribuição final por espécie:")
print(df_sem_outliers['species'].value_counts())


# ==============================================================================
# CONCLUSÃO
# ==============================================================================

print("\n" + "=" * 60)
print("RESUMO FINAL — EDA COMPLETA")
print("=" * 60)

print(f"""
Dataset: Palmer Penguins
Fonte:   seaborn built-in dataset (sns.load_dataset('penguins'))

Etapas realizadas:
  1. Aquisição       → {344} registros carregados
  2. Estrutura       → {df.shape[1]} colunas ({len(colunas_numericas)} numéricas, {len(colunas_categoricas)} categóricas)
  3. Limpeza         → {344 - len(df)} linhas com nulos removidas (dropna)
  4. Univariada      → describe(), value_counts(), histogramas, boxplots
  5. Multivariada    → histogramas/boxplots por espécie, correlação, pairplot
  6. Outliers (IQR)  → {len(df) - len(df_sem_outliers)} outliers removidos
                       Dataset final: {len(df_sem_outliers)} linhas

Gráficos salvos em: {OUTPUT_DIR}
Total de figuras:   {_fig_counter[0]}
""")

print("Concluído!")
