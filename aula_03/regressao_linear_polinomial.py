"""
Machine Learning
Aula 03 - Regressão Linear e Polinomial

Autor: Cláudio Ferreira Neves
Cargo: Analista de BI — Save Co. | Docente SENAI/SC
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Backend não-interativo: salva figuras sem abrir janela
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import (
    PolynomialFeatures,
    StandardScaler,
    OneHotEncoder,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.datasets import fetch_california_housing

# ---------------------------------------------------------------------------
# Pasta de saída para os gráficos
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Contador global de figuras salvas
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


# ==============================================================================
# SEÇÃO 1 — REGRESSÃO LINEAR SIMPLES
# ==============================================================================

print("=" * 60)
print("SEÇÃO 1 — REGRESSÃO LINEAR SIMPLES")
print("=" * 60)

print("""
Regressão Linear Simples modela a relação entre UMA variável
preditora (x) e uma variável alvo contínua (y):

    ŷ = β₀ + β₁x

  β₀ = intercepto (valor de ŷ quando x = 0)
  β₁ = coeficiente angular (variação de ŷ por unidade de x)

Analogia imobiliária: x = metragem (m²), y = preço.
A cada metro quadrado a mais, o preço sobe β₁ unidades.
""")

np.random.seed(42)
n_samples = 100

# Dados artificiais: metragem de 40 a 100 m²
# Preço = 3.5 * metragem + 20 + ruído gaussiano (σ=20)
X_lin = np.linspace(40, 100, n_samples).reshape(-1, 1)
y_lin = 3.5 * X_lin.flatten() + 20 + np.random.normal(0, 20, n_samples)

print(f"Shape X: {X_lin.shape}   Shape y: {y_lin.shape}")
print(f"X range: [{X_lin.min():.0f}, {X_lin.max():.0f}]")
print(f"y range: [{y_lin.min():.1f}, {y_lin.max():.1f}]")

# Divisão treino/teste (80%/20%)
X_tr, X_te, y_tr, y_te = train_test_split(X_lin, y_lin, test_size=0.2, random_state=42)
print(f"\nTreino: {len(X_tr)} amostras   Teste: {len(X_te)} amostras")

# Treinar modelo de Regressão Linear
model_lin = LinearRegression()
model_lin.fit(X_tr, y_tr)

beta_0 = model_lin.intercept_
beta_1 = model_lin.coef_[0]
print(f"\nCoeficientes aprendidos:")
print(f"  β₀ (intercepto) = {beta_0:.4f}")
print(f"  β₁ (angular)    = {beta_1:.4f}")
print(f"  Equação: ŷ = {beta_0:.2f} + {beta_1:.2f} * x")

# Avaliação
y_pred_lin = model_lin.predict(X_te)
r2_lin   = r2_score(y_te, y_pred_lin)
mse_lin  = mean_squared_error(y_te, y_pred_lin)
rmse_lin = np.sqrt(mse_lin)
mae_lin  = mean_absolute_error(y_te, y_pred_lin)

print(f"\nMétricas no conjunto de teste:")
print(f"  R²   = {r2_lin:.4f}")
print(f"  MSE  = {mse_lin:.4f}")
print(f"  RMSE = {rmse_lin:.4f}")
print(f"  MAE  = {mae_lin:.4f}")

# Previsão pontual
x_novo = 75
preco_previsto = model_lin.predict([[x_novo]])[0]
print(f"\nPrevisão para {x_novo} m²: ŷ = {preco_previsto:.2f}")

# Gráfico 1: Scatter + reta ajustada + ponto de previsão
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

x_curve = np.linspace(40, 100, 200).reshape(-1, 1)
axes[0].scatter(X_lin, y_lin, alpha=0.5, color="steelblue", s=25, label="Dados")
axes[0].plot(x_curve, model_lin.predict(x_curve), "r-", linewidth=2,
             label=f"ŷ = {beta_0:.1f} + {beta_1:.2f}x")
axes[0].scatter([x_novo], [preco_previsto], color="orange", s=200,
                marker="*", zorder=5, label=f"Previsão: {preco_previsto:.1f}")
axes[0].set_xlabel("Metragem (m²)")
axes[0].set_ylabel("Preço")
axes[0].set_title("Regressão Linear Simples — Imóveis")
axes[0].legend(fontsize=8)

# Gráfico 2: Paridade
axes[1].scatter(y_te, y_pred_lin, alpha=0.6, color="purple", s=35)
lims = [min(y_te.min(), y_pred_lin.min()), max(y_te.max(), y_pred_lin.max())]
axes[1].plot(lims, lims, "k--", linewidth=1.5, label="Previsão perfeita")
axes[1].set_xlabel("Valor Real")
axes[1].set_ylabel("Valor Previsto")
axes[1].set_title(f"Paridade: Real vs Previsto (R²={r2_lin:.3f})")
axes[1].legend(fontsize=8)

save_fig("lin_simples")


# ==============================================================================
# SEÇÃO 2 — REGRESSÃO LINEAR MÚLTIPLA
# ==============================================================================

print("\n" + "=" * 60)
print("SEÇÃO 2 — REGRESSÃO LINEAR MÚLTIPLA")
print("=" * 60)

print("""
Regressão Linear Múltipla generaliza o modelo para N preditores:

    ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

Cada variável preditora xᵢ contribui com o coeficiente βᵢ.
O modelo minimiza o MSE ajustando todos os β simultaneamente.
""")

# --- 2a) Dados artificiais ---
print("--- 2a) Dados Artificiais: preço = 3.2*metragem + 50*quartos + 30 + ruído ---")

np.random.seed(0)
n_imoveis = 200
metragem = np.random.uniform(40, 200, n_imoveis)
quartos   = np.random.randint(1, 5, n_imoveis).astype(float)
preco     = 3.2 * metragem + 50 * quartos + 30 + np.random.normal(0, 30, n_imoveis)

df_imoveis = pd.DataFrame({"metragem": metragem, "quartos": quartos, "preco": preco})
print(f"\nDataset shape: {df_imoveis.shape}")
print(df_imoveis.head())
print(df_imoveis.describe().round(2))

X_mult = df_imoveis[["metragem", "quartos"]].values
y_mult = df_imoveis["preco"].values
X_tr_m, X_te_m, y_tr_m, y_te_m = train_test_split(
    X_mult, y_mult, test_size=0.2, random_state=42
)

modelo_mult = LinearRegression()
modelo_mult.fit(X_tr_m, y_tr_m)
y_pred_m = modelo_mult.predict(X_te_m)

b0_m = modelo_mult.intercept_
b1_m, b2_m = modelo_mult.coef_
r2_m   = r2_score(y_te_m, y_pred_m)
rmse_m = np.sqrt(mean_squared_error(y_te_m, y_pred_m))
mae_m  = mean_absolute_error(y_te_m, y_pred_m)

print(f"\nEquação aprendida:")
print(f"  ŷ = {b0_m:.2f} + {b1_m:.2f} * metragem + {b2_m:.2f} * quartos")
print(f"\nMétricas: R²={r2_m:.4f}  RMSE={rmse_m:.2f}  MAE={mae_m:.2f}")

# Gráfico múltipla: scatter metragem + scatter quartos + paridade
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].scatter(df_imoveis["metragem"], df_imoveis["preco"],
                alpha=0.4, color="steelblue", s=15)
axes[0].set_xlabel("Metragem (m²)")
axes[0].set_ylabel("Preço")
axes[0].set_title("Metragem × Preço")

for q in sorted(df_imoveis["quartos"].unique()):
    subset = df_imoveis[df_imoveis["quartos"] == q]
    axes[1].scatter(subset["quartos"], subset["preco"], alpha=0.4, s=15,
                    label=f"{int(q)} qtos")
axes[1].set_xlabel("Quartos")
axes[1].set_ylabel("Preço")
axes[1].set_title("Quartos × Preço")
axes[1].legend(fontsize=7)

axes[2].scatter(y_te_m, y_pred_m, alpha=0.6, color="purple", s=25)
lims_m = [min(y_te_m.min(), y_pred_m.min()), max(y_te_m.max(), y_pred_m.max())]
axes[2].plot(lims_m, lims_m, "k--", linewidth=1.5)
axes[2].set_xlabel("Real")
axes[2].set_ylabel("Previsto")
axes[2].set_title(f"Paridade Múltipla (R²={r2_m:.3f})")

save_fig("lin_multipla_artificial")

# --- 2b) Dataset Insurance ---
print("\n--- 2b) Dataset Insurance (charges) ---")

try:
    url_ins = (
        "https://raw.githubusercontent.com/matheusvanzan/"
        "Machine-Learning-Examples/refs/heads/master/datasets/insurance.csv"
    )
    df_ins = pd.read_csv(url_ins).dropna().drop_duplicates().reset_index(drop=True)
    print(f"Dataset Insurance shape: {df_ins.shape}")
    print(df_ins.head())
    print(df_ins.dtypes)

    # Distribuição de charges por fumante
    fig, ax = plt.subplots(figsize=(7, 4))
    for smk, color in zip(["yes", "no"], ["tomato", "steelblue"]):
        ax.hist(df_ins[df_ins["smoker"] == smk]["charges"],
                bins=30, alpha=0.6, color=color, label=f"Fumante: {smk}")
    ax.set_xlabel("Charges (USD)")
    ax.set_ylabel("Frequência")
    ax.set_title("Distribuição de Charges por Hábito de Fumar")
    ax.legend()
    save_fig("insurance_hist_charges")

    num_features = ["age", "bmi", "children"]
    cat_features = ["sex", "smoker", "region"]

    X_ins = df_ins[num_features + cat_features]
    y_ins = df_ins["charges"].values

    X_tr_ins, X_te_ins, y_tr_ins, y_te_ins = train_test_split(
        X_ins, y_ins, test_size=0.2, random_state=42
    )

    # ColumnTransformer + Pipeline
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_features),
    ])

    pipe_ins = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression()),
    ])
    pipe_ins.fit(X_tr_ins, y_tr_ins)  # fit apenas no treino
    y_pred_ins = pipe_ins.predict(X_te_ins)

    r2_ins   = r2_score(y_te_ins, y_pred_ins)
    rmse_ins = np.sqrt(mean_squared_error(y_te_ins, y_pred_ins))
    mae_ins  = mean_absolute_error(y_te_ins, y_pred_ins)

    print(f"\nMétricas Insurance: R²={r2_ins:.4f}  RMSE=${rmse_ins:,.0f}  MAE=${mae_ins:,.0f}")

    # Paridade Insurance
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_te_ins, y_pred_ins, alpha=0.5, color="steelblue", s=20)
    lims_ins = [0, max(y_te_ins.max(), y_pred_ins.max())]
    ax.plot(lims_ins, lims_ins, "k--", linewidth=1.5, label="Perfeito")
    ax.set_xlabel("Charges Real (USD)")
    ax.set_ylabel("Charges Previsto (USD)")
    ax.set_title(f"Paridade — Insurance (R²={r2_ins:.3f})")
    ax.legend(fontsize=8)
    save_fig("insurance_paridade_linear")

except Exception as err:
    print(f"  AVISO: Não foi possível carregar Insurance. {err}")


# ==============================================================================
# SEÇÃO 3 — GRADIENTE DESCENDENTE
# ==============================================================================

print("\n" + "=" * 60)
print("SEÇÃO 3 — GRADIENTE DESCENDENTE")
print("=" * 60)

print("""
O Gradiente Descendente é um algoritmo de otimização que minimiza
a função de custo J(β) de forma iterativa:

    β ← β − η × ∇J(β)

  η = taxa de aprendizado (learning rate)
  ∇J(β) = gradiente da função de custo em relação a β

Para MSE: ∇J(β) = (2/n) * Xᵀ(Xβ − y)

O SGDRegressor do scikit-learn implementa o Gradiente
Descendente Estocástico: atualiza β a cada amostra (ou mini-batch).
""")

# Visualização da função de custo J(β) = (β-2)² + 1
beta_vals = np.linspace(-2, 6, 300)
J_vals = (beta_vals - 2) ** 2 + 1

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax_idx, (eta, label_eta) in enumerate([(0.1, "Passos pequenos η=0.1"),
                                             (0.9, "Passos grandes η=0.9")]):
    beta_t = -1.5
    steps = [beta_t]
    for _ in range(15):
        grad = 2 * (beta_t - 2)
        beta_t = beta_t - eta * grad
        steps.append(beta_t)
        if abs(beta_t) > 30:
            break

    print(f"\n{label_eta}: {len(steps)} passos")
    print(f"  Posições β: {[f'{s:.3f}' for s in steps[:6]]} ...")
    print(f"  β final: {steps[-1]:.6f}  (mínimo em β=2.0)")

    axes[ax_idx].plot(beta_vals, J_vals, color="steelblue", linewidth=2.5,
                      label="J(β) = (β-2)²+1")
    for s in steps:
        j_s = (s - 2) ** 2 + 1
        if -2 <= j_s <= 20:  # manter na janela do gráfico
            axes[ax_idx].scatter([s], [j_s], color="red", s=40, zorder=5)
    axes[ax_idx].scatter([steps[-1]], [(steps[-1] - 2) ** 2 + 1],
                          color="green", s=150, marker="*", zorder=6, label="Final")
    axes[ax_idx].set_xlabel("β")
    axes[ax_idx].set_ylabel("J(β)")
    axes[ax_idx].set_title(label_eta, fontweight="bold")
    axes[ax_idx].legend(fontsize=8)
    axes[ax_idx].set_ylim(-0.5, 20)

save_fig("gradiente_descendente_curva")

# SGDRegressor vs LinearRegression no Insurance
print("\n--- SGDRegressor vs LinearRegression (Insurance) ---")

try:
    df_gd = pd.read_csv(url_ins).dropna().drop_duplicates().reset_index(drop=True)
    num_f = ["age", "bmi", "children"]
    cat_f = ["sex", "smoker", "region"]
    X_gd = df_gd[num_f + cat_f]
    y_gd = df_gd["charges"].values
    X_tr_gd, X_te_gd, y_tr_gd, y_te_gd = train_test_split(
        X_gd, y_gd, test_size=0.2, random_state=42
    )

    for eta0_val, max_iter_val in [(0.01, 500), (0.001, 2000), (0.05, 300)]:
        pre_sgd = ColumnTransformer(transformers=[
            ("num", StandardScaler(), num_f),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_f),
        ])
        pipe_sgd = Pipeline([
            ("pre", pre_sgd),
            ("model", SGDRegressor(eta0=eta0_val, max_iter=max_iter_val,
                                   random_state=42, learning_rate="constant")),
        ])
        pipe_sgd.fit(X_tr_gd, y_tr_gd)
        y_pr_sgd = pipe_sgd.predict(X_te_gd)
        r2_s  = r2_score(y_te_gd, y_pr_sgd)
        rmse_s = np.sqrt(mean_squared_error(y_te_gd, y_pr_sgd))
        print(f"  SGD (η={eta0_val}, iter={max_iter_val}): R²={r2_s:.4f}  RMSE=${rmse_s:,.0f}")

    # Comparar com LinearRegression
    pre_lr2 = ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_f),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_f),
    ])
    pipe_lr2 = Pipeline([("pre", pre_lr2), ("model", LinearRegression())])
    pipe_lr2.fit(X_tr_gd, y_tr_gd)
    y_pr_lr2 = pipe_lr2.predict(X_te_gd)
    r2_lr2  = r2_score(y_te_gd, y_pr_lr2)
    rmse_lr2 = np.sqrt(mean_squared_error(y_te_gd, y_pr_lr2))
    print(f"  LinearRegression (analítico):     R²={r2_lr2:.4f}  RMSE=${rmse_lr2:,.0f}")

except Exception as err:
    print(f"  AVISO: Não foi possível carregar Insurance para SGD. {err}")


# ==============================================================================
# SEÇÃO 4 — REGRESSÃO POLINOMIAL
# ==============================================================================

print("\n" + "=" * 60)
print("SEÇÃO 4 — REGRESSÃO POLINOMIAL")
print("=" * 60)

print("""
A Regressão Polinomial estende a linear adicionando potências de x:

  Grau 1: ŷ = β₀ + β₁x                  (reta)
  Grau 2: ŷ = β₀ + β₁x + β₂x²           (parábola)
  Grau 3: ŷ = β₀ + β₁x + β₂x² + β₃x³   (curva cúbica)

O modelo ainda é LINEAR nos parâmetros β.
Usamos PolynomialFeatures para criar as novas colunas
e LinearRegression para ajustar os coeficientes.

ATENÇÃO: graus muito altos causam overfitting.
""")

# Demonstração de PolynomialFeatures
print("--- Expansão de features com PolynomialFeatures ---")
x_demo = np.array([[2.0], [3.0], [4.0]])
for grau_demo in [2, 3]:
    pf = PolynomialFeatures(degree=grau_demo, include_bias=False)
    x_transformed = pf.fit_transform(x_demo)
    print(f"\n  Grau {grau_demo}: {x_demo.flatten()} → {x_transformed.shape[1]} features")
    print(f"  Colunas: {pf.get_feature_names_out(['x'])}")
    for row_orig, row_trans in zip(x_demo, x_transformed):
        print(f"    {row_orig[0]} → {row_trans}")

# Dados y = 1 - exp(-X) + ruído
np.random.seed(7)
X_poly_raw = np.sort(np.random.uniform(0, 5, 120))
y_poly_raw = 1 - np.exp(-X_poly_raw) + np.random.normal(0, 0.08, 120)
X_poly = X_poly_raw.reshape(-1, 1)
X_tr_p, X_te_p, y_tr_p, y_te_p = train_test_split(
    X_poly, y_poly_raw, test_size=0.2, random_state=42
)

print("\n--- Comparação de modelos por grau ---")
resultados = []
for grau in range(1, 9):
    pipe_p = Pipeline([
        ("poly", PolynomialFeatures(degree=grau, include_bias=False)),
        ("model", LinearRegression()),
    ])
    pipe_p.fit(X_tr_p, y_tr_p)
    y_pr_treino = pipe_p.predict(X_tr_p)
    y_pr_teste  = pipe_p.predict(X_te_p)
    r2_tr = r2_score(y_tr_p, y_pr_treino)
    r2_te = r2_score(y_te_p, y_pr_teste)
    rmse_te = np.sqrt(mean_squared_error(y_te_p, y_pr_teste))
    resultados.append({"grau": grau, "R²_treino": r2_tr, "R²_teste": r2_te, "RMSE_teste": rmse_te})
    print(f"  Grau {grau}: R²_treino={r2_tr:.4f}  R²_teste={r2_te:.4f}  RMSE_teste={rmse_te:.4f}")

df_res = pd.DataFrame(resultados)

# Gráfico comparativo: ajuste visual graus 1, 2, 4
x_curve = np.linspace(0, 5, 300).reshape(-1, 1)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax_i, grau_plot in enumerate([1, 2, 4]):
    pipe_plot = Pipeline([
        ("poly", PolynomialFeatures(degree=grau_plot, include_bias=False)),
        ("model", LinearRegression()),
    ])
    pipe_plot.fit(X_tr_p, y_tr_p)
    y_curve = pipe_plot.predict(x_curve)
    y_pr_te = pipe_plot.predict(X_te_p)
    r2_plot = r2_score(y_te_p, y_pr_te)

    axes[ax_i].scatter(X_poly_raw, y_poly_raw, alpha=0.4, color="steelblue",
                       s=15, label="Dados")
    axes[ax_i].plot(x_curve, y_curve, color="crimson", linewidth=2)
    y_true_curve = 1 - np.exp(-x_curve.flatten())
    axes[ax_i].plot(x_curve, y_true_curve, "g--", linewidth=1.5, alpha=0.7,
                    label="y verdadeiro")
    axes[ax_i].set_xlabel("X")
    axes[ax_i].set_ylabel("y")
    axes[ax_i].set_title(f"Grau {grau_plot} — R²={r2_plot:.3f}")
    axes[ax_i].legend(fontsize=7)

save_fig("poly_comparacao_graus")

# Gráfico de R² treino vs teste por grau (overfitting)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(df_res["grau"], df_res["R²_treino"], "b-o", label="R² treino", linewidth=2)
ax.plot(df_res["grau"], df_res["R²_teste"],  "r-o", label="R² teste",  linewidth=2)
ax.axvline(x=2, color="green", linestyle="--", alpha=0.7, label="Grau ideal ≈ 2")
ax.set_xlabel("Grau do Polinômio")
ax.set_ylabel("R²")
ax.set_title("R² Treino vs Teste por Grau (Overfitting)")
ax.legend()
ax.set_xticks(range(1, 9))
save_fig("poly_overfitting_curve")

print("\nGráfico de overfitting salvo.")


# ==============================================================================
# SEÇÃO 5 — DATASET REAL: CALIFORNIA HOUSING
# ==============================================================================

print("\n" + "=" * 60)
print("SEÇÃO 5 — DATASET REAL: CALIFORNIA HOUSING")
print("=" * 60)

print("""
California Housing Dataset (sklearn):
  - 20640 amostras (blocos censitários da Califórnia, 1990)
  - 8 features: MedInc, HouseAge, AveRooms, AveBedrms,
                Population, AveOccup, Latitude, Longitude
  - Alvo: MedHouseVal — valor mediano dos imóveis ($100.000)
""")

housing = fetch_california_housing(as_frame=True)
df_cal  = housing.frame
feat_names = housing.feature_names

print(f"Shape: {df_cal.shape}")
print(df_cal.head())
print("\nEstatísticas descritivas:")
print(df_cal.describe().round(3))

X_cal = df_cal[list(feat_names)].values
y_cal = df_cal["MedHouseVal"].values

X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
    X_cal, y_cal, test_size=0.2, random_state=42
)
print(f"\nTreino: {len(X_tr_c)}   Teste: {len(X_te_c)}")

# --- 5a) Regressão Linear ---
print("\n--- 5a) Regressão Linear ---")

pipe_cal_lin = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  LinearRegression()),
])
pipe_cal_lin.fit(X_tr_c, y_tr_c)
y_pred_cl = pipe_cal_lin.predict(X_te_c)

r2_cl   = r2_score(y_te_c, y_pred_cl)
rmse_cl = np.sqrt(mean_squared_error(y_te_c, y_pred_cl))
mae_cl  = mean_absolute_error(y_te_c, y_pred_cl)
print(f"LinearRegression: R²={r2_cl:.4f}  RMSE={rmse_cl:.4f}  MAE={mae_cl:.4f}")
print(f"  (RMSE em dólares: ~${rmse_cl * 100000:,.0f})")

# Paridade linear
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(y_te_c, y_pred_cl, alpha=0.2, color="steelblue", s=8)
lims_c = [y_te_c.min(), y_te_c.max()]
ax.plot(lims_c, lims_c, "k--", linewidth=1.5, label="Perfeito")
ax.set_xlabel("MedHouseVal Real ($100k)")
ax.set_ylabel("MedHouseVal Previsto ($100k)")
ax.set_title(f"Paridade — California Housing Linear (R²={r2_cl:.3f})")
ax.legend(fontsize=8)
save_fig("california_linear_paridade")

# --- 5b) Regressão Polinomial (grau 2) ---
print("\n--- 5b) Regressão Polinomial (grau 2) ---")

pipe_cal_poly = Pipeline([
    ("scaler", StandardScaler()),
    ("poly",   PolynomialFeatures(degree=2, include_bias=False)),
    ("model",  LinearRegression()),
])
pipe_cal_poly.fit(X_tr_c, y_tr_c)
y_pred_cp = pipe_cal_poly.predict(X_te_c)

r2_cp   = r2_score(y_te_c, y_pred_cp)
rmse_cp = np.sqrt(mean_squared_error(y_te_c, y_pred_cp))
mae_cp  = mean_absolute_error(y_te_c, y_pred_cp)
print(f"PolynomialFeatures(grau=2): R²={r2_cp:.4f}  RMSE={rmse_cp:.4f}  MAE={mae_cp:.4f}")
print(f"  (RMSE em dólares: ~${rmse_cp * 100000:,.0f})")
print(f"  Melhora R²: {(r2_cp - r2_cl) * 100:+.2f} pontos percentuais")

# Paridade polinomial
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(y_te_c, y_pred_cp, alpha=0.2, color="seagreen", s=8)
ax.plot(lims_c, lims_c, "k--", linewidth=1.5, label="Perfeito")
ax.set_xlabel("MedHouseVal Real ($100k)")
ax.set_ylabel("MedHouseVal Previsto ($100k)")
ax.set_title(f"Paridade — California Housing Poly grau 2 (R²={r2_cp:.3f})")
ax.legend(fontsize=8)
save_fig("california_poly_paridade")

# --- 5c) Tabela comparativa ---
print("\n--- 5c) Tabela comparativa ---")
df_final = pd.DataFrame({
    "Modelo":  ["LinearRegression", "PolynomialFeatures(grau=2)"],
    "R²":      [round(r2_cl, 4),   round(r2_cp, 4)],
    "RMSE":    [round(rmse_cl, 4), round(rmse_cp, 4)],
    "MAE":     [round(mae_cl, 4),  round(mae_cp, 4)],
})
print(df_final.to_string(index=False))

# Importâncias dos coeficientes (modelo linear)
coefs_lin = pipe_cal_lin.named_steps["model"].coef_
print("\nCoeficientes LinearRegression (features padronizadas):")
for name, coef in sorted(zip(feat_names, coefs_lin), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {name:<14}: {coef:+.4f}")

# Gráfico de coeficientes
fig, ax = plt.subplots(figsize=(8, 5))
sorted_idx = np.argsort(np.abs(coefs_lin))[::-1]
colors_bar = ["tomato" if c < 0 else "steelblue" for c in coefs_lin[sorted_idx]]
ax.barh([feat_names[i] for i in sorted_idx], coefs_lin[sorted_idx], color=colors_bar)
ax.axvline(x=0, color="black", linewidth=0.8)
ax.set_xlabel("Coeficiente (features padronizadas)")
ax.set_title("Importância dos Coeficientes — California Housing Linear")
save_fig("california_coeficientes")

print("\n" + "=" * 60)
print("FIM DO SCRIPT — Todos os gráficos salvos em outputs/")
print("=" * 60)
