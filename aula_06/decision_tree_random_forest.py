"""
Machine Learning
Aula 06 - Decision Tree e Random Forest

Autor: Cláudio Ferreira Neves
Cargo: Analista de BI — Save Co. | Docente SENAI/SC
Certificação: DATA ANALYST CERTIFIED PROFESSIONAL © (DACP)

Script didático: executa todos os exemplos e salva figuras em outputs/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Backend não-interativo: salva figuras sem abrir janela
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Pasta de saída para os gráficos
# ---------------------------------------------------------------------------
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Contador global de figuras salvas
_fig_counter = [0]


def save_fig(name: str):
    """Salva a figura atual em outputs/ com numeração sequencial e 150 DPI."""
    _fig_counter[0] += 1
    filename = f"{_fig_counter[0]:02d}_{name}.png"
    filepath = os.path.join(OUTPUTS_DIR, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Figura salva] {filename}")


def banner(titulo: str):
    """Imprime um banner de seção formatado."""
    linha = "=" * 68
    print(f"\n{linha}")
    print(f"  {titulo}")
    print(linha)


def sub_banner(titulo: str):
    """Imprime um sub-banner de seção."""
    linha = "-" * 60
    print(f"\n{linha}")
    print(f"  {titulo}")
    print(linha)


# ==============================================================================
# SEÇÃO 1 — DECISION TREE (Breast Cancer)
# ==============================================================================

banner("SEÇÃO 1 — DECISION TREE | Dataset: Breast Cancer")

print("""
Árvore de Decisão (Decision Tree)
──────────────────────────────────
Uma árvore de decisão divide recursivamente o espaço de features
em regiões cada vez mais puras, aplicando um critério de impureza.

Critério Gini:
    Gini(t) = 1 - Σ p(c|t)²
    → mede a probabilidade de classificar erroneamente uma amostra
      escolhida aleatoriamente (0 = nó puro, 0.5 = máxima impureza)

Information Gain (Entropia):
    H(t) = - Σ p(c|t) · log₂ p(c|t)
    IG = H(pai) - Σ [|filho| / |pai|] · H(filho)
    → quanto maior o IG, mais informativa é a divisão

Parâmetros-chave:
  max_depth      → profundidade máxima (controla overfitting)
  min_samples_split → mínimo de amostras para dividir um nó
  criterion      → 'gini' ou 'entropy'
""")

# --- Carregar dataset ---
sub_banner("Carregando Breast Cancer Dataset")
bc = load_breast_cancer()
X_bc = pd.DataFrame(bc.data, columns=bc.feature_names)
y_bc = pd.Series(bc.target, name="target")

print(f"Amostras  : {X_bc.shape[0]}")
print(f"Features  : {X_bc.shape[1]}")
print(f"Classes   : {list(bc.target_names)} (0=malignant, 1=benign)")
print(f"Distribuição:\n{y_bc.value_counts().to_string()}")

# --- Divisão treino/teste ---
X_train, X_test, y_train, y_test = train_test_split(
    X_bc, y_bc, test_size=0.2, random_state=42, stratify=y_bc
)
print(f"\nTreino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")

# --- Pipeline Decision Tree ---
sub_banner("Treinando Decision Tree (criterion='gini', max_depth=10)")
pipeline_dt = Pipeline([
    ("clf", DecisionTreeClassifier(criterion="gini", max_depth=10, random_state=42))
])
pipeline_dt.fit(X_train, y_train)
y_pred_dt = pipeline_dt.predict(X_test)

print("\nClassification Report — Decision Tree:")
print(classification_report(y_test, y_pred_dt, target_names=bc.target_names))

acc_dt = accuracy_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt, average="macro")
print(f"Acurácia : {acc_dt:.4f}")
print(f"F1-Macro : {f1_dt:.4f}")

# --- Visualização da árvore ---
sub_banner("Visualizando a árvore (max_depth=2 para legibilidade)")
fig_tree, ax_tree = plt.subplots(figsize=(14, 8))
plot_tree(
    pipeline_dt.named_steps["clf"],
    feature_names=bc.feature_names,
    class_names=bc.target_names,
    filled=True,
    max_depth=2,
    ax=ax_tree,
    fontsize=9,
)
ax_tree.set_title("Decision Tree — Breast Cancer (exibindo até profundidade 2)", fontsize=13, fontweight="bold")
save_fig("decision_tree_estrutura")

# --- Matriz de confusão DT ---
fig_cm_dt, ax_cm_dt = plt.subplots(figsize=(6, 5))
cm_dt = confusion_matrix(y_test, y_pred_dt)
sns.heatmap(cm_dt, annot=True, fmt="d", cmap="Blues",
            xticklabels=bc.target_names, yticklabels=bc.target_names,
            ax=ax_cm_dt)
ax_cm_dt.set_title("Confusion Matrix — Decision Tree", fontweight="bold")
ax_cm_dt.set_ylabel("Real")
ax_cm_dt.set_xlabel("Previsto")
save_fig("decision_tree_confusion_matrix")

# --- Cross-validation DT ---
sub_banner("Cross-Validation (cv=10) — Decision Tree")
cv_dt = cross_validate(
    pipeline_dt, X_bc, y_bc, cv=10,
    scoring=["accuracy", "f1_macro"],
    return_train_score=False,
)
print(f"Acurácia  CV10: {cv_dt['test_accuracy'].mean():.4f} ± {cv_dt['test_accuracy'].std():.4f}")
print(f"F1-Macro  CV10: {cv_dt['test_f1_macro'].mean():.4f} ± {cv_dt['test_f1_macro'].std():.4f}")

# --- Impacto de max_depth ---
sub_banner("Análise: Acurácia vs max_depth (1 a 20)")
depths = range(1, 21)
acc_train_list, acc_test_list = [], []
for d in depths:
    pipe_tmp = Pipeline([("clf", DecisionTreeClassifier(max_depth=d, random_state=42))])
    pipe_tmp.fit(X_train, y_train)
    acc_train_list.append(accuracy_score(y_train, pipe_tmp.predict(X_train)))
    acc_test_list.append(accuracy_score(y_test, pipe_tmp.predict(X_test)))

fig_depth, ax_depth = plt.subplots(figsize=(10, 5))
ax_depth.plot(list(depths), acc_train_list, marker="o", label="Treino", color="#667eea")
ax_depth.plot(list(depths), acc_test_list, marker="s", label="Teste", color="#e53e3e")
ax_depth.axvline(x=10, color="gray", linestyle="--", alpha=0.6, label="max_depth=10 (usado)")
ax_depth.set_xlabel("max_depth")
ax_depth.set_ylabel("Acurácia")
ax_depth.set_title("Acurácia vs max_depth — Decision Tree", fontweight="bold")
ax_depth.legend()
ax_depth.grid(True, alpha=0.3)
save_fig("decision_tree_depth_vs_accuracy")

print("\nOBSERVAÇÃO: Árvores muito profundas memorizam o treino (overfitting).")
print("  → Acurácia de treino sobe para 1.0, mas teste piora ou estagna.")


# ==============================================================================
# SEÇÃO 2 — RANDOM FOREST (Breast Cancer)
# ==============================================================================

banner("SEÇÃO 2 — RANDOM FOREST | Dataset: Breast Cancer")

print("""
Random Forest — Ensemble de Árvores
─────────────────────────────────────
Random Forest combina N árvores de decisão independentes treinadas em
subconjuntos aleatórios de dados (bootstrap) e features.

Conceitos fundamentais:
  Bagging (Bootstrap Aggregating):
    → Cada árvore é treinada em uma amostra com reposição do conjunto treino.
    → Reduz a variância sem aumentar o viés.

  Seleção aleatória de features:
    → Em cada nó, apenas sqrt(n_features) features são consideradas.
    → Garante diversidade entre as árvores.

  Out-of-Bag Error (OOB):
    → ~37% das amostras não são usadas no treino de cada árvore.
    → Permitem estimar o erro sem necessidade de validação separada.

  Votação majoritária:
    → A classe final é a mais votada entre todas as árvores.
""")

# --- Pipeline Random Forest ---
sub_banner("Treinando Random Forest (n_estimators=10, criterion='gini')")
pipeline_rf = Pipeline([
    ("clf", RandomForestClassifier(n_estimators=10, criterion="gini", random_state=42))
])
pipeline_rf.fit(X_train, y_train)
y_pred_rf = pipeline_rf.predict(X_test)

print("\nClassification Report — Random Forest:")
print(classification_report(y_test, y_pred_rf, target_names=bc.target_names))

acc_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf, average="macro")
print(f"Acurácia : {acc_rf:.4f}")
print(f"F1-Macro : {f1_rf:.4f}")

# --- Matriz de confusão RF ---
fig_cm_rf, ax_cm_rf = plt.subplots(figsize=(6, 5))
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens",
            xticklabels=bc.target_names, yticklabels=bc.target_names,
            ax=ax_cm_rf)
ax_cm_rf.set_title("Confusion Matrix — Random Forest", fontweight="bold")
ax_cm_rf.set_ylabel("Real")
ax_cm_rf.set_xlabel("Previsto")
save_fig("random_forest_confusion_matrix")

# --- Cross-validation RF ---
sub_banner("Cross-Validation (cv=10) — Random Forest")
cv_rf = cross_validate(
    pipeline_rf, X_bc, y_bc, cv=10,
    scoring=["accuracy", "f1_macro"],
    return_train_score=False,
)
print(f"Acurácia  CV10: {cv_rf['test_accuracy'].mean():.4f} ± {cv_rf['test_accuracy'].std():.4f}")
print(f"F1-Macro  CV10: {cv_rf['test_f1_macro'].mean():.4f} ± {cv_rf['test_f1_macro'].std():.4f}")

# --- Impacto de n_estimators ---
sub_banner("Análise: Acurácia vs n_estimators (5 a 100)")
n_est_list = list(range(5, 101, 5))
acc_rf_list = []
for n in n_est_list:
    pipe_tmp = Pipeline([("clf", RandomForestClassifier(n_estimators=n, random_state=42))])
    pipe_tmp.fit(X_train, y_train)
    acc_rf_list.append(accuracy_score(y_test, pipe_tmp.predict(X_test)))

fig_nest, ax_nest = plt.subplots(figsize=(10, 5))
ax_nest.plot(n_est_list, acc_rf_list, marker="o", color="#38a169")
ax_nest.set_xlabel("n_estimators")
ax_nest.set_ylabel("Acurácia (Teste)")
ax_nest.set_title("Acurácia vs n_estimators — Random Forest", fontweight="bold")
ax_nest.grid(True, alpha=0.3)
save_fig("random_forest_estimators_vs_accuracy")

# --- Comparação DT vs RF ---
sub_banner("Comparação DT vs RF (métricas lado a lado)")
print(f"{'Modelo':<20} {'Acurácia':>10} {'F1-Macro':>10}")
print("-" * 42)
print(f"{'Decision Tree':<20} {acc_dt:>10.4f} {f1_dt:>10.4f}")
print(f"{'Random Forest':<20} {acc_rf:>10.4f} {f1_rf:>10.4f}")


# ==============================================================================
# SEÇÃO 3 — XGBOOST (Breast Cancer)
# ==============================================================================

banner("SEÇÃO 3 — XGBOOST | Dataset: Breast Cancer")

print("""
XGBoost — eXtreme Gradient Boosting
──────────────────────────────────────
Boosting é uma técnica de ensemble SEQUENCIAL: cada árvore corrige
os erros da anterior, ao contrário do Bagging que é paralelo.

Conceitos:
  Gradient Boosting:
    → Treina modelos sequencialmente minimizando uma função de perda.
    → Cada nova árvore é ajustada nos resíduos (erros) do modelo atual.

  Learning Rate (η):
    → Escala a contribuição de cada nova árvore.
    → Valores menores = aprendizado mais lento, mas mais robusto.

  Regularização:
    → lambda (L2) e alpha (L1) penalizam modelos complexos.
    → Evita sobreajuste melhor que árvores isoladas.

  XGBoost vs Random Forest:
    → RF: paralelo, reduz variância
    → XGB: sequencial, reduz viés e variância com regularização

ATENÇÃO: XGBoost requer target numérico inteiro. Usar LabelEncoder.
""")

# --- Preparar target para XGBoost ---
le = LabelEncoder()
y_bc_xgb = le.fit_transform(y_bc)
X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
    X_bc, y_bc_xgb, test_size=0.2, random_state=42, stratify=y_bc_xgb
)

# --- Pipeline XGBoost básico ---
sub_banner("Treinando XGBoost (parâmetros padrão)")
pipeline_xgb = Pipeline([
    ("clf", XGBClassifier(eval_metric="logloss", random_state=42, verbosity=0))
])
pipeline_xgb.fit(X_train_xgb, y_train_xgb)
y_pred_xgb = pipeline_xgb.predict(X_test_xgb)

print("\nClassification Report — XGBoost (básico):")
print(classification_report(y_test_xgb, y_pred_xgb, target_names=bc.target_names))

acc_xgb = accuracy_score(y_test_xgb, y_pred_xgb)
f1_xgb = f1_score(y_test_xgb, y_pred_xgb, average="macro")
print(f"Acurácia : {acc_xgb:.4f}")
print(f"F1-Macro : {f1_xgb:.4f}")

# --- Matriz de confusão XGB ---
fig_cm_xgb, ax_cm_xgb = plt.subplots(figsize=(6, 5))
cm_xgb = confusion_matrix(y_test_xgb, y_pred_xgb)
sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Oranges",
            xticklabels=bc.target_names, yticklabels=bc.target_names,
            ax=ax_cm_xgb)
ax_cm_xgb.set_title("Confusion Matrix — XGBoost", fontweight="bold")
ax_cm_xgb.set_ylabel("Real")
ax_cm_xgb.set_xlabel("Previsto")
save_fig("xgboost_confusion_matrix")

# --- Cross-validation XGB ---
sub_banner("Cross-Validation (cv=10) — XGBoost")
cv_xgb = cross_validate(
    pipeline_xgb, X_bc, y_bc_xgb, cv=10,
    scoring=["accuracy", "f1_macro"],
    return_train_score=False,
)
print(f"Acurácia  CV10: {cv_xgb['test_accuracy'].mean():.4f} ± {cv_xgb['test_accuracy'].std():.4f}")
print(f"F1-Macro  CV10: {cv_xgb['test_f1_macro'].mean():.4f} ± {cv_xgb['test_f1_macro'].std():.4f}")

# --- XGBoost com parâmetros avançados ---
sub_banner("XGBoost com parâmetros avançados")
pipeline_xgb_adv = Pipeline([
    ("clf", XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    ))
])
pipeline_xgb_adv.fit(X_train_xgb, y_train_xgb)
y_pred_xgb_adv = pipeline_xgb_adv.predict(X_test_xgb)
acc_xgb_adv = accuracy_score(y_test_xgb, y_pred_xgb_adv)
f1_xgb_adv = f1_score(y_test_xgb, y_pred_xgb_adv, average="macro")
print(f"Acurácia (avançado): {acc_xgb_adv:.4f}")
print(f"F1-Macro (avançado): {f1_xgb_adv:.4f}")


# ==============================================================================
# SEÇÃO 4 — IMPORTÂNCIA DE FEATURES
# ==============================================================================

banner("SEÇÃO 4 — IMPORTÂNCIA DE FEATURES")

print("""
Importância de Features (Feature Importance)
─────────────────────────────────────────────
Cada modelo baseado em árvores calcula a importância das features
com base na redução de impureza (Gini importance):

  Importância = Σ (redução de Gini ponderada por amostras no nó)
                ao longo de todas as divisões que usam a feature.

Interpretação:
  → Features com maior importância contribuem mais para a decisão.
  → Permite simplificar modelos e explicar predições.
  → Cuidado: features de alta cardinalidade podem ser superestimadas.

Permutation Importance:
  → Alternativa mais robusta: embaralha a feature e mede queda de acc.
  → Menos sensível ao tipo de feature (numérica vs categórica).
""")

# Retreinar DT e RF com dados completos para importância consistente
dt_clf = DecisionTreeClassifier(criterion="gini", max_depth=10, random_state=42)
dt_clf.fit(X_train, y_train)

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

xgb_clf = XGBClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=4,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss", random_state=42, verbosity=0
)
xgb_clf.fit(X_train_xgb, y_train_xgb)

feature_names = list(bc.feature_names)

# --- Top-10 importâncias de cada modelo ---
fig_imp, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, clf, title, color in zip(
    axes,
    [dt_clf, rf_clf, xgb_clf],
    ["Decision Tree", "Random Forest", "XGBoost"],
    ["#667eea", "#38a169", "#e67e22"],
):
    importances = pd.Series(clf.feature_importances_, index=feature_names)
    top10 = importances.nlargest(10).sort_values()
    ax.barh(top10.index, top10.values, color=color, alpha=0.85)
    ax.set_title(f"Top-10 Features\n{title}", fontweight="bold", fontsize=11)
    ax.set_xlabel("Importância (Gini)")
    ax.grid(True, axis="x", alpha=0.3)

fig_imp.suptitle("Importância de Features — DT vs RF vs XGBoost", fontsize=14, fontweight="bold")
save_fig("feature_importances_comparacao")

# Imprimir tabela top-5 por modelo
sub_banner("Top-5 Features por Modelo")
for clf, nome in [(dt_clf, "Decision Tree"), (rf_clf, "Random Forest"), (xgb_clf, "XGBoost")]:
    importances = pd.Series(clf.feature_importances_, index=feature_names)
    top5 = importances.nlargest(5)
    print(f"\n{nome}:")
    for feat, val in top5.items():
        print(f"  {feat:<35} {val:.4f}")


# ==============================================================================
# SEÇÃO 5 — COMPARAÇÃO E GRIDSEARCH
# ==============================================================================

banner("SEÇÃO 5 — COMPARAÇÃO GERAL E GRIDSEARCHCV")

# --- Comparação DT vs RF vs XGB (CV) ---
sub_banner("Comparação CV10 — DT vs RF vs XGBoost (Breast Cancer)")

resultados = {}
for nome, pipe, X_data, y_data in [
    ("Decision Tree", pipeline_dt, X_bc, y_bc),
    ("Random Forest", pipeline_rf, X_bc, y_bc),
    ("XGBoost",       pipeline_xgb, X_bc, y_bc_xgb),
]:
    cv_res = cross_validate(pipe, X_data, y_data, cv=10,
                            scoring=["accuracy", "f1_macro"])
    resultados[nome] = {
        "acc_mean": cv_res["test_accuracy"].mean(),
        "acc_std":  cv_res["test_accuracy"].std(),
        "f1_mean":  cv_res["test_f1_macro"].mean(),
        "f1_std":   cv_res["test_f1_macro"].std(),
    }

print(f"\n{'Modelo':<20} {'Acurácia (CV10)':>20} {'F1-Macro (CV10)':>20}")
print("-" * 62)
for nome, res in resultados.items():
    print(
        f"{nome:<20} "
        f"{res['acc_mean']:.4f} ± {res['acc_std']:.4f}   "
        f"{res['f1_mean']:.4f} ± {res['f1_std']:.4f}"
    )

# --- Gráfico de barras agrupado ---
fig_comp, ax_comp = plt.subplots(figsize=(10, 6))
modelos = list(resultados.keys())
x = np.arange(len(modelos))
width = 0.35

acc_means = [resultados[m]["acc_mean"] for m in modelos]
f1_means  = [resultados[m]["f1_mean"]  for m in modelos]
acc_stds  = [resultados[m]["acc_std"]  for m in modelos]
f1_stds   = [resultados[m]["f1_std"]   for m in modelos]

bars1 = ax_comp.bar(x - width/2, acc_means, width, yerr=acc_stds,
                    label="Acurácia", color="#667eea", alpha=0.85,
                    capsize=4, error_kw={"ecolor": "#333", "linewidth": 1.5})
bars2 = ax_comp.bar(x + width/2, f1_means, width, yerr=f1_stds,
                    label="F1-Macro", color="#38a169", alpha=0.85,
                    capsize=4, error_kw={"ecolor": "#333", "linewidth": 1.5})

ax_comp.set_xticks(x)
ax_comp.set_xticklabels(modelos)
ax_comp.set_ylim(0.85, 1.02)
ax_comp.set_ylabel("Score")
ax_comp.set_title("Comparação de Modelos — Breast Cancer (CV=10)", fontweight="bold")
ax_comp.legend()
ax_comp.grid(True, axis="y", alpha=0.3)

for bar in list(bars1) + list(bars2):
    h = bar.get_height()
    ax_comp.annotate(f"{h:.3f}",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8)

save_fig("comparacao_modelos_barras")

# --- GridSearchCV no Credit Dataset ---
sub_banner("GridSearchCV — Dataset Credit Approval (credit-g)")

print("\nCarregando dataset credit-g (OpenML)...")
try:
    credit = fetch_openml("credit-g", version=1, as_frame=True)
    df_credit = credit.frame
    X_credit = credit.data.copy()
    y_credit_raw = credit.target

    # Codificar target
    le_credit = LabelEncoder()
    y_credit = le_credit.fit_transform(y_credit_raw)

    # Identificar colunas categóricas e numéricas
    cat_cols = X_credit.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_credit.select_dtypes(include=[np.number]).columns.tolist()

    print(f"Amostras    : {X_credit.shape[0]}")
    print(f"Features    : {X_credit.shape[1]}")
    print(f"  Numéricas : {len(num_cols)}")
    print(f"  Categ.    : {len(cat_cols)}")
    print(f"Classes     : {list(le_credit.classes_)}")

    # Pré-processador
    preprocessor = ColumnTransformer(transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
    ])

    # param_grid conforme especificado
    param_grid = [
        {
            "clf": [DecisionTreeClassifier(random_state=42)],
            "clf__max_depth": [10, 20, 30],
        },
        {
            "clf": [RandomForestClassifier(random_state=42)],
            "clf__n_estimators": [10, 30, 50],
            "clf__max_depth": [10, 20],
        },
        {
            "clf": [XGBClassifier(eval_metric="logloss", random_state=42, verbosity=0)],
            "clf__n_estimators": [10, 30, 50],
            "clf__max_depth": [10, 20],
        },
    ]

    pipe_gs = Pipeline([
        ("pre", preprocessor),
        ("clf", DecisionTreeClassifier()),
    ])

    print("\nExecutando GridSearchCV (cv=5, scoring='accuracy')...")
    print("  (Isso pode levar alguns minutos...)")

    gs = GridSearchCV(
        pipe_gs, param_grid, cv=5, scoring="accuracy",
        refit=True, n_jobs=-1, verbose=0
    )
    gs.fit(X_credit, y_credit)

    print(f"\nMelhor modelo : {type(gs.best_estimator_.named_steps['clf']).__name__}")
    print(f"Melhores params: {gs.best_params_}")
    print(f"Melhor score  : {gs.best_score_:.4f}")

    # Resumo dos top-5 resultados
    results_df = pd.DataFrame(gs.cv_results_)
    top5 = results_df.nlargest(5, "mean_test_score")[
        ["param_clf", "mean_test_score", "std_test_score"]
    ].copy()
    top5["Modelo"] = top5["param_clf"].apply(lambda x: type(x).__name__)
    top5 = top5.drop(columns=["param_clf"])
    print("\nTop-5 configurações:")
    print(top5.to_string(index=False))

except Exception as e:
    print(f"\n[AVISO] Não foi possível carregar/executar credit-g: {e}")
    print("  Verifique a conexão com a internet e o pacote scikit-learn.")


# ==============================================================================
# RESUMO FINAL
# ==============================================================================

banner("RESUMO — QUANDO USAR CADA MODELO")

print("""
Quando usar cada modelo:
─────────────────────────
  Decision Tree:
    ✓ Quando a interpretabilidade é crucial (visualizar a árvore)
    ✓ Dados com poucas features e relações não-lineares simples
    ✗ Evitar quando há muitos dados ou features — overfitting fácil
    ✗ Sensível a pequenas variações nos dados

  Random Forest:
    ✓ Melhor generalização que DT isolada (menos variância)
    ✓ Robusto a outliers e dados faltantes
    ✓ Boa estimativa de importância de features
    ✗ Mais lento para predição que DT única
    ✗ Menos interpretável que DT única

  XGBoost:
    ✓ Alta performance em tabular data (padrão em competições)
    ✓ Regularização nativa evita overfitting
    ✓ Suporte a dados esparsos e missing values
    ✗ Mais hiperparâmetros para ajustar
    ✗ Mais lento no treino que RF (sequencial)

Regra prática:
  1. Comece com Random Forest → boa linha de base
  2. Se precisar de performance máxima → XGBoost com tuning
  3. Se precisar explicar o modelo → Decision Tree com max_depth limitado
""")

print(f"\n{'='*68}")
print(f"  Aula 06 concluída! Figuras salvas em: {OUTPUTS_DIR}")
print(f"  Total de figuras geradas: {_fig_counter[0]}")
print(f"{'='*68}\n")
