"""
Machine Learning
Aula 01 - Introdução ao Machine Learning
Autor: Cláudio Ferreira Neves
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error


# ==============================================================================
# REVISÃO: REPRESENTAÇÃO DE DADOS NO PLANO
# ==============================================================================

print("=" * 60)
print("REVISÃO: PONTOS NO PLANO E GRÁFICO DE DISPERSÃO")
print("=" * 60)

np.random.seed(42)

print("\nnp.random.rand(5, 2):")
print(np.random.rand(5, 2))

print("\nnp.random.randn(5, 2):")
print(np.random.randn(5, 2))

n = 20
points = np.random.rand(n, 2) * 10
labels = np.random.randint(0, 3, n)

print("\npoints[:10]:", points[:10])
print("labels[:10]:", labels[:10])

# Diferença entre formas de indexação
print("\npoints[:,0] (1D):", points[:, 0].shape)
print("points[:,[0]] (2D):", points[:, [0]].shape)
print("points[:,0].reshape(-1,1) (2D):", points[:, 0].reshape(-1, 1).shape)

# Scatter plot simples
plt.figure(figsize=(5, 5))
plt.scatter(points[:, 0], points[:, 1], c="blue", alpha=0.7)
plt.title("Pontos no Plano (sem classes)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

# Scatter plot com classes (colormap)
plt.figure(figsize=(5, 5))
plt.scatter(points[:, 0], points[:, 1], c=labels, cmap="Set1", alpha=0.7)
plt.title("Pontos no Plano (com classes - cmap Set1)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.colorbar(label="Classe")
plt.show()


# ==============================================================================
# REGRESSÃO LINEAR SIMPLES
# ==============================================================================

print("\n" + "=" * 60)
print("REGRESSÃO LINEAR SIMPLES")
print("=" * 60)

np.random.seed(42)
samples = 100

# Dados artificiais: y = 4 + 3x + ruído
X = 2 * np.random.rand(samples, 1)
e = np.random.randn(samples, 1)
y = 4 + 3 * X + e

print(f"\nFormato X: {X.shape}, y: {y.shape}")
print(f"X[:3]:\n{X[:3]}")
print(f"y[:3]:\n{y[:3]}")

# Visualização dos dados
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color="blue", label="Dados reais")
plt.title("Exemplo de Regressão Linear - Dados")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# --- Solução Manual (Mínimos Quadrados) ---
print("\n--- Solução Manual ---")
mean_x = np.mean(X)
mean_y = np.mean(y)
print(f"mean_x: {mean_x:.4f}, mean_y: {mean_y:.4f}")

numer = np.sum((X - mean_x) * (y - mean_y))
denom = np.sum((X - mean_x) ** 2)

b1 = numer / denom
b0 = mean_y - b1 * mean_x

print(f"b1 (coeficiente): {b1:.4f}")
print(f"b0 (intercepto):  {b0:.4f}")

y_pred_manual = b0 + b1 * X

plt.figure(figsize=(8, 6))
plt.scatter(X, y, color="blue", label="Dados reais")
plt.plot(X, y_pred_manual, color="green", label="Reta manual")
plt.title("Regressão Linear - Solução Manual")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# --- Solução com scikit-learn ---
print("\n--- Solução com scikit-learn ---")
model = LinearRegression()
model.fit(X, y)

b0_sk = model.intercept_[0]
b1_sk = model.coef_[0][0]
print(f"b1 sklearn: {b1_sk:.4f}")
print(f"b0 sklearn: {b0_sk:.4f}")

# Previsão para novo ponto
novo_x = np.array([[1.3]])
novo_y = model.predict(novo_x)
print(f"\nPrevisão: x={novo_x[0][0]:.1f} → y={novo_y[0][0]:.4f}")

y_pred = model.predict(X)

# Comparação: manual vs sklearn
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color="blue", label="Dados reais")
plt.plot(X, y_pred_manual, color="green", label="Reta manual")
plt.plot(X, y_pred, color="red", linewidth=2, label="Regressão Linear (sklearn)")
plt.title("Regressão Linear - Manual vs sklearn")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# Previsão com novo ponto destacado
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color="blue", label="Dados reais")
plt.plot(X, y_pred, color="red", linewidth=2, label="Regressão Linear")
plt.scatter(novo_x, novo_y, color="yellow", edgecolors="k", s=150, marker="*",
            label=f"Previsão para x={novo_x[0][0]:.1f}")
plt.title("Regressão Linear com Previsão")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# --- Métricas ---
print("\n--- Métricas ---")
r2 = model.score(X, y)
print(f"Coeficiente de Determinação (R²): {r2:.4f}")

mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
print(f"Raiz do Erro Quadrático Médio (RMSE): {rmse:.4f}")


# ==============================================================================
# REGRESSÃO POLINOMIAL
# ==============================================================================

print("\n" + "=" * 60)
print("REGRESSÃO POLINOMIAL")
print("=" * 60)

np.random.seed(42)
samples = 100

# Dados: y = 1 - e^(-x) + ruído
X = np.random.rand(samples, 1) * 10
e = np.random.randn(samples, 1) * 0.05
y = 1 - np.exp(-X) + e

print(f"X[:3]:\n{X[:3]}")
print(f"y[:3]:\n{y[:3]}")

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color="blue", label="Dados reais")
plt.title("Dados para Regressão Polinomial")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# Pipeline: PolynomialFeatures + LinearRegression
degree = 2
modelo_polinomial = make_pipeline(
    PolynomialFeatures(degree=degree),
    LinearRegression()
)
modelo_polinomial.fit(X, y)
print(f"\nModelo polinomial (grau={degree}) treinado.")

X_pred = np.linspace(X.min(), X.max(), samples).reshape(-1, 1)
y_poly = modelo_polinomial.predict(X_pred)

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color="blue", label="Dados reais")
plt.plot(X_pred, y_poly, color="red", linewidth=2, label=f"Modelo polinomial (grau={degree})")
plt.title("Regressão Não Linear (curva convergente)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()


# ==============================================================================
# CLASSIFICAÇÃO
# ==============================================================================

print("\n" + "=" * 60)
print("CLASSIFICAÇÃO (REGRESSÃO LOGÍSTICA)")
print("=" * 60)

np.random.seed(42)

# Criação dos dados artificiais
X0 = np.random.randn(100, 2) + np.array([2, 2])   # Classe 0 (azul)
y0 = np.zeros(100)

X1 = np.random.randn(100, 2) + np.array([6, 6])   # Classe 1 (vermelho)
y1 = np.ones(100)

print(f"X0[:3]:\n{X0[:3]}")
print(f"X1[:3]:\n{X1[:3]}")

# Visualização separada por classe
plt.figure(figsize=(8, 6))
plt.scatter(X0[:, 0], X0[:, 1], c="blue", alpha=0.7, label="Classe 0")
plt.scatter(X1[:, 0], X1[:, 1], c="red", alpha=0.7, label="Classe 1")
plt.title("Dataset de Classificação - 2D")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

# Unindo em um único dataset
X = np.vstack((X0, X1))
y = np.hstack((y0, y1))

plt.figure(figsize=(8, 5))
plot = plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", alpha=0.7)
plt.title("Dataset de Classificação - 2D (unificado)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend(handles=plot.legend_elements()[0], labels=["Classe 0", "Classe 1"])
plt.show()

# --- Treinamento ---
model = LogisticRegression()
model.fit(X, y)
print(f"\nModelo treinado: {model}")


def plot_decision_boundary(model, X, y):
    """Plota a fronteira de decisão do classificador."""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 5))
    plt.contourf(xx, yy, Z, cmap="bwr", alpha=0.2)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k")
    plt.title("Classificação com Fronteira de Decisão")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")


# Fronteira de decisão
plot_decision_boundary(model, X, y)
plt.show()

# --- Previsão de novo ponto ---
new_point = [4, 5]
pred = model.predict([new_point])[0]
print(f"\nNovo ponto {new_point} → Classe prevista: {int(pred)}")

plot_decision_boundary(model, X, y)
plt.scatter(new_point[0], new_point[1], c="yellow", edgecolors="k", s=150, marker="*",
            label=f"Novo ponto → Classe {int(pred)}")
plt.legend()
plt.show()


# ==============================================================================
# DIVISÃO TREINO E TESTE
# ==============================================================================

print("\n" + "=" * 60)
print("DIVISÃO TREINO E TESTE")
print("=" * 60)

np.random.seed(42)
X = np.random.randn(100, 2) + np.array([2, 2])
y = np.zeros(100)

print(f"Dataset: X={X.shape}, y={y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Treino: X={X_train.shape}, y={y_train.shape}")
print(f"Teste:  X={X_test.shape}, y={y_test.shape}")

plt.figure(figsize=(8, 5))
plt.scatter(X_train[:, 0], X_train[:, 1], color="blue", label="Dados treino")
plt.scatter(X_test[:, 0], X_test[:, 1], color="red", label="Dados teste")
plt.title("Divisão Treino e Teste")
plt.legend()
plt.show()


# ==============================================================================
# DATASET IRIS — EXEMPLO COMPLETO
# ==============================================================================

print("\n" + "=" * 60)
print("DATASET IRIS — PIPELINE COMPLETO DE MACHINE LEARNING")
print("=" * 60)

# 1. Carregar dataset
iris = datasets.load_iris()
X = iris.data   # (150, 4): sepal length, sepal width, petal length, petal width
y = iris.target # (150,): 0=Setosa, 1=Versicolor, 2=Virginica

print(f"\nFeatures: {iris.feature_names}")
print(f"Classes:  {iris.target_names}")
print(f"X[:3]:\n{X[:3]}")
print(f"y[:3]: {y[:3]}")

# 2. Visualização — Sépala
X_vis = X[:, :2]
plt.figure(figsize=(8, 5))
sct = plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y, cmap="Set1", edgecolor="k")
plt.xlabel("Sepal length (cm)")
plt.ylabel("Sepal width (cm)")
plt.title("Dataset IRIS — Sépala")
handles, _ = sct.legend_elements()
plt.legend(handles, iris.target_names)
plt.show()

# Visualização — Pétala
X_vis = X[:, 2:4]
plt.figure(figsize=(8, 5))
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y, cmap="Set1", edgecolor="k")
plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.title("Dataset IRIS — Pétala")
plt.show()

# 3. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"\nTreino: {X_train.shape} | Teste: {X_test.shape}")

# 4. Treinar modelo
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
print(f"Modelo treinado: {model}")

# 5. Previsões
y_pred = model.predict(X_test)
print(f"\ny_pred[:10]: {y_pred[:10]}")

# 6. Avaliação
print("\n--- Métricas de Avaliação ---")

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")

report = classification_report(y_test, y_pred, target_names=iris.target_names)
print("\nClassification Report:")
print(report)

cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{cm}")

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.title("Matriz de Confusão — IRIS")
plt.tight_layout()
plt.show()

# 7. Fronteira de decisão (2 features)
X_vis = X[:, :2]  # sepal length e sepal width
X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(
    X_vis, y, test_size=0.3, random_state=42, stratify=y
)

model_vis = LogisticRegression(max_iter=200)
model_vis.fit(X_train_vis, y_train_vis)

plot_decision_boundary(model_vis, X_vis, y)
plt.xlabel("Sepal length (cm)")
plt.ylabel("Sepal width (cm)")
plt.title("Classificação IRIS — Regressão Logística (2 features)")
plt.show()

print("\nConcluído!")
