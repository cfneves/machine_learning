"""
=============================================================================
Machine Learning — Aula 04
Regressão Logística e KNN

Autor       : Cláudio Ferreira Neves
Cargo atual : Analista de BI — Save Co. / Especialista de Ensino II — SENAI/SC
Certificação: DATA ANALYST CERTIFIED PROFESSIONAL © (DACP)

Script de linha de comando — executa todas as seções sequencialmente,
imprime resultados no terminal e salva gráficos em outputs/.
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Backend não-interativo: salva figuras sem abrir janela
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

# ---------------------------------------------------------------------------
# Pasta de saída para os gráficos
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

_fig_counter = [0]


def save_fig(name: str):
    """Salva a figura atual em outputs/ com numeração sequencial e 150 DPI."""
    _fig_counter[0] += 1
    filename = f"{_fig_counter[0]:02d}_{name}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  [Figura salva] {filename}")


def banner(titulo: str):
    """Imprime um banner de seção no terminal."""
    linha = "=" * 70
    print(f"\n{linha}")
    print(f"  {titulo}")
    print(linha)


# ==============================================================================
# SEÇÃO 1 — FUNÇÃO SIGMOID E REGRESSÃO LOGÍSTICA (TEORIA)
# ==============================================================================

banner("SEÇÃO 1 — FUNÇÃO SIGMOID E REGRESSÃO LOGÍSTICA")

print("""
A Regressão Logística NÃO é um algoritmo de regressão no sentido usual.
Ela usa a função sigmoid para converter uma combinação linear de features
em uma PROBABILIDADE entre 0 e 1.

  Combinação linear:  z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
  Função Sigmoid:     σ(z) = 1 / (1 + e^(-z))
  Regra de decisão:   classe = 1  se  σ(z) ≥ 0.5
                      classe = 0  se  σ(z) < 0.5

Diferença fundamental em relação à Regressão Linear:
  • Linear:   saída é um número contínuo (preço, temperatura...)
  • Logística: saída é uma probabilidade [0, 1] → classificação binária

Função de custo: Log-Loss (Binary Cross-Entropy)
  J = -1/n Σ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]

Log-odds (logit): log(p / (1-p)) = z
  A cada incremento de 1 em xⱼ, as odds são multiplicadas por e^(βⱼ).
""")

# Demonstração numérica da sigmoid
z_values = np.linspace(-10, 10, 1000)
sigmoid = 1 / (1 + np.exp(-z_values))

print("Valores da sigmoid em pontos específicos:")
for z_pt in [-5, -2, -1, 0, 1, 2, 5]:
    s_pt = 1 / (1 + np.exp(-z_pt))
    print(f"  σ({z_pt:+3d}) = {s_pt:.6f}")

# Gráfico sigmoid
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(z_values, sigmoid, color="#667eea", linewidth=2.5, label="σ(z) = 1/(1+e⁻ᶻ)")
ax.axhline(0.5, color="tomato", linestyle="--", linewidth=1.5, label="Limiar 0.5")
ax.axvline(0, color="gray", linestyle=":", linewidth=1, alpha=0.7)
ax.fill_between(z_values, sigmoid, 0.5, where=(sigmoid >= 0.5),
                alpha=0.15, color="seagreen", label="Classe 1")
ax.fill_between(z_values, sigmoid, 0.5, where=(sigmoid < 0.5),
                alpha=0.15, color="tomato", label="Classe 0")
ax.set_xlabel("z (combinação linear)")
ax.set_ylabel("σ(z) — probabilidade")
ax.set_title("Função Sigmoid — Regressão Logística")
ax.legend(fontsize=9)
ax.set_ylim(-0.05, 1.05)
ax.grid(alpha=0.3)
save_fig("sigmoid_funcao")


# ==============================================================================
# SEÇÃO 2 — DATASET DIABETES (REGRESSÃO LOGÍSTICA BINÁRIA)
# ==============================================================================

banner("SEÇÃO 2 — DATASET DIABETES (REGRESSÃO LOGÍSTICA BINÁRIA)")

print("""
Dataset: Pima Indians Diabetes Database
  768 pacientes do sexo feminino, com 21+ anos de origem Pima.
  Target: Outcome (0 = sem diabetes, 1 = com diabetes)

Features:
  Pregnancies, Glucose, BloodPressure, SkinThickness,
  Insulin, BMI, DiabetesPedigreeFunction, Age

Importante: valores 0 em colunas clínicas (Glucose, BloodPressure,
SkinThickness, Insulin, BMI) são biologicamente impossíveis — tratamos
como valores ausentes e imputamos com a mediana via SimpleImputer.
""")

URL_DIABETES = (
    "https://raw.githubusercontent.com/matheusvanzan/"
    "Machine-Learning-Examples/refs/heads/master/datasets/diabetes.csv"
)

try:
    df_diab = pd.read_csv(URL_DIABETES)
    print(f"Shape: {df_diab.shape}")
    print(f"Colunas: {list(df_diab.columns)}")
    print(f"\nBalanceamento do target:")
    print(df_diab["Outcome"].value_counts())
    print(f"\nPorcentagem de positivos: {df_diab['Outcome'].mean()*100:.1f}%")
    print("\nEstatísticas descritivas:")
    print(df_diab.describe().round(2).to_string())

    # Contagem de zeros (valores ausentes implícitos) nas colunas clínicas
    cols_impute = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    print("\nContagem de zeros por coluna clínica (serão imputados):")
    for col in cols_impute:
        n_zeros = (df_diab[col] == 0).sum()
        pct = n_zeros / len(df_diab) * 100
        print(f"  {col:<25}: {n_zeros:3d} zeros ({pct:.1f}%)")

    # Histograma de Glucose por Outcome
    fig, ax = plt.subplots(figsize=(8, 5))
    for outcome, color, label in [(0, "steelblue", "Sem diabetes"), (1, "tomato", "Com diabetes")]:
        ax.hist(df_diab[df_diab["Outcome"] == outcome]["Glucose"],
                bins=30, alpha=0.6, color=color, label=label)
    ax.set_xlabel("Glucose")
    ax.set_ylabel("Frequência")
    ax.set_title("Distribuição de Glicose por Diagnóstico de Diabetes")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig("diabetes_glucose_hist")

    # Heatmap de correlação
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = df_diab.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.5)
    ax.set_title("Mapa de Correlação — Diabetes")
    save_fig("diabetes_correlacao")

    # Preparação
    X_diab = df_diab.drop("Outcome", axis=1)
    y_diab = df_diab["Outcome"]
    feat_names_diab = X_diab.columns.tolist()
    num_all = feat_names_diab

    X_tr_d, X_te_d, y_tr_d, y_te_d = train_test_split(
        X_diab, y_diab, test_size=0.2, random_state=42, stratify=y_diab
    )
    print(f"\nTreino: {len(X_tr_d)}  Teste: {len(X_te_d)}")

    preprocessor_diab = ColumnTransformer(transformers=[
        ("imputer_scaler", Pipeline([
            ("imp", SimpleImputer(missing_values=0, strategy="median")),
            ("scl", StandardScaler()),
        ]), cols_impute),
        ("scaler_only", StandardScaler(),
         [c for c in feat_names_diab if c not in cols_impute]),
    ])

    pipe_diab = Pipeline([
        ("pre", preprocessor_diab),
        ("clf", LogisticRegression(penalty="l2", max_iter=1000, random_state=42)),
    ])
    pipe_diab.fit(X_tr_d, y_tr_d)
    y_pred_d = pipe_diab.predict(X_te_d)

    print("\n--- Relatório de Classificação — Diabetes ---")
    print(classification_report(y_te_d, y_pred_d, target_names=["Sem Diabetes", "Com Diabetes"]))

    # Matriz de confusão
    cm_d = confusion_matrix(y_te_d, y_pred_d)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_d, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Sem Diabetes", "Com Diabetes"],
                yticklabels=["Sem Diabetes", "Com Diabetes"])
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão — Diabetes (Regressão Logística)")
    save_fig("diabetes_confusion_matrix_lr")

    # Cross-validation
    print("\n--- Validação Cruzada (cv=10) — Diabetes ---")
    cv_results = cross_validate(
        pipe_diab, X_diab, y_diab, cv=10,
        scoring=["accuracy", "f1_macro"], return_train_score=False
    )
    acc_mean = cv_results["test_accuracy"].mean()
    acc_std  = cv_results["test_accuracy"].std()
    f1_mean  = cv_results["test_f1_macro"].mean()
    f1_std   = cv_results["test_f1_macro"].std()
    print(f"  Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"  F1-macro: {f1_mean:.4f} ± {f1_std:.4f}")

except Exception as err:
    print(f"  AVISO: Não foi possível carregar o dataset Diabetes. {err}")
    df_diab = None
    pipe_diab = None


# ==============================================================================
# SEÇÃO 3 — DATASET PENGUINS (REGRESSÃO LOGÍSTICA MULTI-CLASSE)
# ==============================================================================

banner("SEÇÃO 3 — DATASET PENGUINS (REGRESSÃO LOGÍSTICA MULTI-CLASSE)")

print("""
Dataset: Palmer Penguins
  344 pinguins de 3 espécies: Adelie, Chinstrap, Gentoo
  Target: espécie (3 classes) — problema multi-classe

Estratégia multi-classe da Regressão Logística:
  • One-vs-Rest (OvR): treina um classificador binário por classe.
    "Adelie vs não-Adelie", "Chinstrap vs não-Chinstrap", etc.
    A classe com maior probabilidade vence.
  • Multinomial (Softmax): treina em todas as classes simultaneamente.
    Extensão natural para K classes via função softmax.

Pre-processamento:
  • Numéricas: StandardScaler
  • Categóricas (ilha, sexo): OneHotEncoder(drop='first')
""")

try:
    df_pen = sns.load_dataset("penguins")

    # Renomear colunas para português
    df_pen.columns = [
        "espécie", "ilha", "comprimento_bico_mm", "profundidade_bico_mm",
        "comprimento_nadadeira_mm", "massa_corporal_g", "sexo"
    ]
    # Adicionar coluna ano se vier (versão sem ano no seaborn)
    if "ano" not in df_pen.columns and len(df_pen.columns) == 7:
        pass  # seaborn não inclui 'year' por padrão

    # Mapear sexo
    df_pen["sexo"] = df_pen["sexo"].map({"Male": "macho", "Female": "fêmea"})
    df_pen = df_pen.dropna().reset_index(drop=True)

    print(f"Shape após dropna: {df_pen.shape}")
    print(f"Espécies: {df_pen['espécie'].value_counts().to_dict()}")
    print(f"Ilhas: {df_pen['ilha'].value_counts().to_dict()}")
    print(df_pen.head())

    # Pairplot
    fig = sns.pairplot(df_pen, hue="espécie", palette="tab10",
                       vars=["comprimento_bico_mm", "profundidade_bico_mm",
                             "comprimento_nadadeira_mm", "massa_corporal_g"],
                       plot_kws={"alpha": 0.5, "s": 20})
    fig.fig.suptitle("Pairplot — Penguins por Espécie", y=1.01, fontsize=13)
    path_pair = os.path.join(OUTPUT_DIR, f"{_fig_counter[0]+1:02d}_penguins_pairplot.png")
    _fig_counter[0] += 1
    fig.savefig(path_pair, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  [Figura salva] {os.path.basename(path_pair)}")

    # Correlação numérica
    num_cols_pen = ["comprimento_bico_mm", "profundidade_bico_mm",
                    "comprimento_nadadeira_mm", "massa_corporal_g"]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(df_pen[num_cols_pen].corr(), annot=True, fmt=".2f",
                cmap="coolwarm", center=0, ax=ax, linewidths=0.5)
    ax.set_title("Correlação Numérica — Penguins")
    save_fig("penguins_correlacao")

    # Preparação
    X_pen = df_pen.drop("espécie", axis=1)
    y_pen = df_pen["espécie"]
    num_pen = ["comprimento_bico_mm", "profundidade_bico_mm",
               "comprimento_nadadeira_mm", "massa_corporal_g"]
    cat_pen = ["ilha", "sexo"]

    X_tr_p, X_te_p, y_tr_p, y_te_p = train_test_split(
        X_pen, y_pen, test_size=0.2, stratify=y_pen, random_state=25
    )

    pre_pen = ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_pen),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_pen),
    ])
    pipe_pen = Pipeline([
        ("pre", pre_pen),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipe_pen.fit(X_tr_p, y_tr_p)
    y_pred_p = pipe_pen.predict(X_te_p)

    print("\n--- Relatório de Classificação — Penguins ---")
    print(classification_report(y_te_p, y_pred_p))

    # Matriz de confusão Penguins
    cm_p = confusion_matrix(y_te_p, y_pred_p, labels=pipe_pen.classes_)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_p, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=pipe_pen.classes_,
                yticklabels=pipe_pen.classes_)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão — Penguins (Regressão Logística)")
    save_fig("penguins_confusion_matrix_lr")

    # Cross-validation
    print("\n--- Validação Cruzada (cv=10) — Penguins ---")
    cv_pen = cross_validate(
        pipe_pen, X_pen, y_pen, cv=10,
        scoring=["accuracy", "f1_macro"], return_train_score=False
    )
    print(f"  Accuracy: {cv_pen['test_accuracy'].mean():.4f} ± {cv_pen['test_accuracy'].std():.4f}")
    print(f"  F1-macro: {cv_pen['test_f1_macro'].mean():.4f} ± {cv_pen['test_f1_macro'].std():.4f}")

except Exception as err:
    print(f"  AVISO: Não foi possível carregar o dataset Penguins. {err}")
    df_pen = None
    pipe_pen = None


# ==============================================================================
# SEÇÃO 4 — KNN — K-NEAREST NEIGHBORS
# ==============================================================================

banner("SEÇÃO 4 — KNN — K-NEAREST NEIGHBORS")

print("""
KNN — K-Vizinhos Mais Próximos

Algoritmo preguiçoso (lazy learner): não constrói um modelo explícito.
Na predição, calcula a distância do novo ponto para todos os exemplos
de treino e atribui a classe majoritária entre os k mais próximos.

Distância Euclidiana (a mais comum):
  d(p, q) = √( Σ (pᵢ - qᵢ)² )

Outras métricas: Manhattan (L1), Minkowski, Cosine...

Hiperparâmetro k:
  • k pequeno (k=1): baixo viés, alta variância → overfitting
  • k grande     : alto viés, baixa variância → underfitting

Maldição da dimensionalidade:
  Em alta dimensão, todos os pontos ficam igualmente "distantes",
  tornando o conceito de vizinhança menos informativo.
  Solução: padronização das features + redução de dimensionalidade (PCA).
""")

# Demonstração visual KNN com dados sintéticos
np.random.seed(99)
n_demo = 40
X_demo_0 = np.random.randn(n_demo, 2) + np.array([-1.5, 0])
X_demo_1 = np.random.randn(n_demo, 2) + np.array([1.5, 0])
X_demo   = np.vstack([X_demo_0, X_demo_1])
y_demo   = np.array([0] * n_demo + [1] * n_demo)
novo_ponto = np.array([[0.2, 0.5]])

for k_demo in [1, 3, 5, 7]:
    from sklearn.neighbors import KNeighborsClassifier as KNC
    knn_demo = KNC(n_neighbors=k_demo)
    knn_demo.fit(X_demo, y_demo)
    pred_demo = knn_demo.predict(novo_ponto)[0]
    proba_demo = knn_demo.predict_proba(novo_ponto)[0]

    # Criar meshgrid para fronteira de decisão
    h = 0.05
    x_min, x_max = X_demo[:, 0].min() - 1, X_demo[:, 0].max() + 1
    y_min, y_max = X_demo[:, 1].min() - 1, X_demo[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = knn_demo.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.contourf(xx, yy, Z, alpha=0.2, cmap="RdBu")
    ax.scatter(X_demo_0[:, 0], X_demo_0[:, 1], color="steelblue", s=35,
               alpha=0.7, label="Classe 0")
    ax.scatter(X_demo_1[:, 0], X_demo_1[:, 1], color="tomato", s=35,
               alpha=0.7, label="Classe 1")
    ax.scatter(novo_ponto[0, 0], novo_ponto[0, 1],
               color="gold", s=200, marker="*", zorder=5,
               label=f"Novo ponto → Classe {pred_demo}")

    # Destacar os k vizinhos mais próximos
    dists = np.sqrt(((X_demo - novo_ponto) ** 2).sum(axis=1))
    idx_vizinhos = np.argsort(dists)[:k_demo]
    for idx_v in idx_vizinhos:
        ax.plot([novo_ponto[0, 0], X_demo[idx_v, 0]],
                [novo_ponto[0, 1], X_demo[idx_v, 1]],
                "k--", linewidth=0.8, alpha=0.6)

    ax.set_title(f"KNN Demostração — k={k_demo} | Predição: Classe {pred_demo} "
                 f"(P={proba_demo[pred_demo]:.2f})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save_fig(f"knn_demo_k{k_demo}")
    print(f"  k={k_demo}: predito=Classe {pred_demo}  proba={proba_demo}")


# --- KNN no Diabetes ---
print("\n--- KNN (k=3) no Dataset Diabetes ---")
try:
    pipe_knn_d = Pipeline([
        ("pre", ColumnTransformer(transformers=[
            ("imputer_scaler", Pipeline([
                ("imp", SimpleImputer(missing_values=0, strategy="median")),
                ("scl", StandardScaler()),
            ]), ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]),
            ("scaler_only", StandardScaler(),
             ["Pregnancies", "DiabetesPedigreeFunction", "Age"]),
        ])),
        ("clf", KNeighborsClassifier(n_neighbors=3)),
    ])
    X_diab_full = df_diab.drop("Outcome", axis=1)
    y_diab_full = df_diab["Outcome"]
    X_tr_kd, X_te_kd, y_tr_kd, y_te_kd = train_test_split(
        X_diab_full, y_diab_full, test_size=0.2, random_state=42, stratify=y_diab_full
    )
    pipe_knn_d.fit(X_tr_kd, y_tr_kd)
    y_pred_kd = pipe_knn_d.predict(X_te_kd)
    acc_kd = accuracy_score(y_te_kd, y_pred_kd)
    f1_kd  = f1_score(y_te_kd, y_pred_kd, average="macro")
    print(f"  Accuracy: {acc_kd:.4f}  F1-macro: {f1_kd:.4f}")
    print(classification_report(y_te_kd, y_pred_kd,
                                 target_names=["Sem Diabetes", "Com Diabetes"]))

    cm_kd = confusion_matrix(y_te_kd, y_pred_kd)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_kd, annot=True, fmt="d", cmap="Oranges", ax=ax,
                xticklabels=["Sem Diabetes", "Com Diabetes"],
                yticklabels=["Sem Diabetes", "Com Diabetes"])
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão — Diabetes (KNN k=3)")
    save_fig("diabetes_confusion_matrix_knn")

except Exception as err:
    print(f"  AVISO: {err}")
    acc_kd, f1_kd = None, None

# --- KNN no Penguins ---
print("\n--- KNN (k=5) no Dataset Penguins ---")
try:
    pipe_knn_p = Pipeline([
        ("pre", ColumnTransformer(transformers=[
            ("num", StandardScaler(), num_pen),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_pen),
        ])),
        ("clf", KNeighborsClassifier(n_neighbors=5)),
    ])
    pipe_knn_p.fit(X_tr_p, y_tr_p)
    y_pred_kp = pipe_knn_p.predict(X_te_p)
    acc_kp = accuracy_score(y_te_p, y_pred_kp)
    f1_kp  = f1_score(y_te_p, y_pred_kp, average="macro")
    print(f"  Accuracy: {acc_kp:.4f}  F1-macro: {f1_kp:.4f}")
    print(classification_report(y_te_p, y_pred_kp))

    cm_kp = confusion_matrix(y_te_p, y_pred_kp, labels=pipe_pen.classes_)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_kp, annot=True, fmt="d", cmap="Oranges", ax=ax,
                xticklabels=pipe_pen.classes_,
                yticklabels=pipe_pen.classes_)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão — Penguins (KNN k=5)")
    save_fig("penguins_confusion_matrix_knn")

except Exception as err:
    print(f"  AVISO: {err}")
    acc_kp, f1_kp = None, None

# --- Tabela comparativa LR vs KNN ---
print("\n--- Comparação: Regressão Logística vs KNN ---")
try:
    acc_lr_d = accuracy_score(y_te_d, y_pred_d)
    f1_lr_d  = f1_score(y_te_d, y_pred_d, average="macro")
    acc_lr_p = accuracy_score(y_te_p, y_pred_p)
    f1_lr_p  = f1_score(y_te_p, y_pred_p, average="macro")

    df_comp = pd.DataFrame({
        "Dataset":   ["Diabetes", "Diabetes", "Penguins", "Penguins"],
        "Modelo":    ["LogisticRegression", "KNN (k=3)",
                      "LogisticRegression", "KNN (k=5)"],
        "Accuracy":  [round(acc_lr_d, 4), round(acc_kd, 4) if acc_kd else "—",
                      round(acc_lr_p, 4), round(acc_kp, 4) if acc_kp else "—"],
        "F1-macro":  [round(f1_lr_d, 4), round(f1_kd, 4) if f1_kd else "—",
                      round(f1_lr_p, 4), round(f1_kp, 4) if f1_kp else "—"],
    })
    print(df_comp.to_string(index=False))
except Exception as err:
    print(f"  AVISO: {err}")


# ==============================================================================
# SEÇÃO 5 — OTIMIZAÇÃO DO K (BIAS-VARIANCE TRADEOFF)
# ==============================================================================

banner("SEÇÃO 5 — OTIMIZAÇÃO DO K")

print("""
Como escolher o melhor k?

  • k muito pequeno (k=1): modelo memoriza os dados de treino (overfitting).
    Alta variância — pequenas mudanças nos dados mudam muito o resultado.

  • k muito grande: modelo ignora a estrutura local dos dados (underfitting).
    Alto viés — decisões baseadas numa vizinhança muito ampla.

  • k ótimo: equilíbrio entre viés e variância.
    Geralmente encontrado com cross-validation no conjunto de treino.

Dica: testar k ímpar evita empates em classificação binária.
""")

# --- Otimização K no Diabetes ---
print("--- Curva Accuracy e F1 vs K — Dataset Diabetes ---")
try:
    k_range = range(3, 301, 10)
    accs_d, f1s_d = [], []

    for k_val in k_range:
        pipe_k = Pipeline([
            ("pre", ColumnTransformer(transformers=[
                ("imputer_scaler", Pipeline([
                    ("imp", SimpleImputer(missing_values=0, strategy="median")),
                    ("scl", StandardScaler()),
                ]), ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]),
                ("scaler_only", StandardScaler(),
                 ["Pregnancies", "DiabetesPedigreeFunction", "Age"]),
            ])),
            ("clf", KNeighborsClassifier(n_neighbors=k_val)),
        ])
        cv_k = cross_validate(
            pipe_k, X_diab_full, y_diab_full, cv=5,
            scoring=["accuracy", "f1_macro"]
        )
        accs_d.append(cv_k["test_accuracy"].mean())
        f1s_d.append(cv_k["test_f1_macro"].mean())

    best_k_acc_d = list(k_range)[np.argmax(accs_d)]
    best_k_f1_d  = list(k_range)[np.argmax(f1s_d)]
    print(f"  Melhor k por Accuracy: {best_k_acc_d} ({max(accs_d):.4f})")
    print(f"  Melhor k por F1-macro: {best_k_f1_d} ({max(f1s_d):.4f})")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(list(k_range), accs_d, "b-o", markersize=4, linewidth=2, label="Accuracy")
    ax.plot(list(k_range), f1s_d,  "r-s", markersize=4, linewidth=2, label="F1-macro")
    ax.axvline(best_k_acc_d, color="blue", linestyle="--", alpha=0.6,
               label=f"Melhor k (Acc) = {best_k_acc_d}")
    ax.axvline(best_k_f1_d, color="red", linestyle="--", alpha=0.6,
               label=f"Melhor k (F1) = {best_k_f1_d}")
    ax.set_xlabel("k (n_neighbors)")
    ax.set_ylabel("Métrica (cv=5)")
    ax.set_title("Otimização do K — Dataset Diabetes")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    save_fig("otimizacao_k_diabetes")

except Exception as err:
    print(f"  AVISO: {err}")

# --- Otimização K no Penguins ---
print("\n--- Curva Accuracy e F1 vs K — Dataset Penguins ---")
try:
    accs_p, f1s_p = [], []

    for k_val in k_range:
        pipe_kp = Pipeline([
            ("pre", ColumnTransformer(transformers=[
                ("num", StandardScaler(), num_pen),
                ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_pen),
            ])),
            ("clf", KNeighborsClassifier(n_neighbors=k_val)),
        ])
        cv_kp = cross_validate(
            pipe_kp, X_pen, y_pen, cv=5,
            scoring=["accuracy", "f1_macro"]
        )
        accs_p.append(cv_kp["test_accuracy"].mean())
        f1s_p.append(cv_kp["test_f1_macro"].mean())

    best_k_acc_p = list(k_range)[np.argmax(accs_p)]
    best_k_f1_p  = list(k_range)[np.argmax(f1s_p)]
    print(f"  Melhor k por Accuracy: {best_k_acc_p} ({max(accs_p):.4f})")
    print(f"  Melhor k por F1-macro: {best_k_f1_p} ({max(f1s_p):.4f})")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(list(k_range), accs_p, "b-o", markersize=4, linewidth=2, label="Accuracy")
    ax.plot(list(k_range), f1s_p,  "r-s", markersize=4, linewidth=2, label="F1-macro")
    ax.axvline(best_k_acc_p, color="blue", linestyle="--", alpha=0.6,
               label=f"Melhor k (Acc) = {best_k_acc_p}")
    ax.axvline(best_k_f1_p, color="red", linestyle="--", alpha=0.6,
               label=f"Melhor k (F1) = {best_k_f1_p}")
    ax.set_xlabel("k (n_neighbors)")
    ax.set_ylabel("Métrica (cv=5)")
    ax.set_title("Otimização do K — Dataset Penguins")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    save_fig("otimizacao_k_penguins")

except Exception as err:
    print(f"  AVISO: {err}")


# ==============================================================================
# FIM DO SCRIPT
# ==============================================================================

print("\n" + "=" * 70)
print("  FIM DO SCRIPT — Todos os gráficos salvos em outputs/")
print("=" * 70)
