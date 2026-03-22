# 🔮 Aula 04 — Regressão Logística e KNN

<div align="center">

**Cláudio Ferreira Neves  ·  Especialista em Ciência de Dados e IA**

</div>

> Material didático autoral. Reprodução ou uso sem crédito ao autor é considerado plágio.

---

## O que é essa aula?

Até aqui você aprendeu a prever números — preços, custos, temperaturas. Mas e quando a resposta não é um número, e sim uma categoria? "Esse paciente tem diabetes ou não?" "Esse passageiro sobreviveu ao Titanic?" "Qual espécie é esse pinguim?" Esse tipo de problema se chama **classificação**, e é onde a maioria das aplicações práticas de Machine Learning vive.

Nesta aula você vai aprender dois algoritmos de classificação bem diferentes entre si: a **Regressão Logística** (que, apesar do nome, classifica — não regride) e o **KNN** (K-Nearest Neighbors, K Vizinhos Mais Próximos). Entender as duas abordagens dá uma base boa para encarar qualquer problema de classificação.

Os datasets desta aula são reais: **Diabetes** (diagnóstico médico), **Penguins** (classificação de espécies) e **Titanic** (sobrevivência a bordo).

---

## Como o projeto está organizado

```
aula_04/
│
├── app_streamlit.py                    # Aplicação web interativa — abre no navegador
├── aula04_logistica_knn.py             # Script Python principal — roda no terminal
├── requirements.txt                    # Bibliotecas necessárias
├── README.md                           # Este arquivo
│
└── outputs/                            # Gráficos salvos automaticamente (150 DPI)
```

> A pasta `outputs/` é criada automaticamente na primeira execução.

---

## Como rodar na sua máquina

### 1. Instale as dependências

```bash
pip install -r requirements.txt
```

Ou manualmente:

```bash
pip install streamlit numpy pandas matplotlib seaborn scikit-learn
```

### 2. Abra a aplicação no navegador

```bash
streamlit run app_streamlit.py --server.port 8504
```

Acesse `http://localhost:8504` no navegador. Se estiver usando o portal principal, a navegação entre aulas já está configurada.

---

## O que você vai aprender — 4 seções

| # | Seção | O que é trabalhado |
|---|-------|--------------------|
| 1 | **Regressão Logística — Teoria** | A função sigmoid, como ela transforma qualquer número em probabilidade, e por que isso resolve classificação |
| 2 | **Regressão Logística — Prática** | Diagnóstico de Diabetes (binário) e classificação de espécies de Pinguins (multi-classe) com Pipeline completo, validação cruzada e matriz de confusão |
| 3 | **KNN — Teoria e Fronteira de Decisão** | Distância Euclidiana, como o algoritmo vota com os vizinhos, e visualização interativa da fronteira de decisão |
| 4 | **KNN — Prática e Comparativo** | KNN nos datasets Diabetes e Penguins, curva de Accuracy × K para encontrar o K ideal, e comparação direta com Regressão Logística |

---

## O que dá pra mexer em tempo real

- **Separação treino/teste** → veja como mais dados de treino melhoram os resultados
- **Número de folds (cv)** → controle a validação cruzada estratificada
- **Valor de K no KNN** → explore como o número de vizinhos afeta as fronteiras de decisão e as métricas
- **Demonstração 2D interativa** → adicione um ponto novo e veja o KNN classificando em tempo real
- **Curvas de desempenho** → acompanhe a evolução de Accuracy e F1 à medida que K cresce

---

## Os conceitos principais

### Regressão Logística

Apesar do nome "regressão", ela é usada para classificar. O truque é usar a **função sigmoid** (em forma de S) para transformar qualquer número em uma probabilidade entre 0 e 1:

$$P(y=1 \mid X) = \sigma(z) = \frac{1}{1 + e^{-z}} \quad \text{onde} \quad z = \beta_0 + \boldsymbol{\beta}^T X$$

Se a probabilidade for maior que 0,5 → classe 1. Menor que 0,5 → classe 0. Simples assim.

Para classificação com mais de duas classes (multi-classe), o scikit-learn usa automaticamente a estratégia **OvR (One-vs-Rest, Um contra o Resto)**: treina um classificador binário para cada classe e escolhe a que tiver maior probabilidade.

### KNN — K Vizinhos Mais Próximos

A ideia é intuitiva: para classificar um ponto novo, o KNN olha para os **K pontos mais próximos** no conjunto de treino e vota na classe mais frequente.

A distância mais comum é a **Euclidiana**:

$$d(p, q) = \sqrt{\sum_{i=1}^{n}(p_i - q_i)^2}$$

O KNN é sensível à escala das variáveis. Uma variável que vai de 0 a 1.000 vai dominar outra que vai de 0 a 1, então use o `StandardScaler` antes de aplicar o algoritmo.

Quanto à escolha de K: valores muito pequenos fazem o modelo memorizar os dados (overfitting); valores muito grandes fazem ele ignorar detalhes importantes (underfitting). A curva de Accuracy × K mostra onde está o ponto de equilíbrio.

---

## Validação Cruzada Estratificada

Avaliar um modelo nos mesmos dados em que ele treinou produz resultados otimistas. A **validação cruzada (cross-validation)** divide os dados em K grupos, treina em K-1 e testa no restante, rotacionando até todos os grupos terem sido usados como teste.

A versão **estratificada** garante que a proporção de cada classe seja mantida em cada fold — essencial quando os dados são desbalanceados (por exemplo, muito mais "sem diabetes" do que "com diabetes").

```python
from sklearn.model_selection import cross_validate, StratifiedKFold

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
resultados = cross_validate(pipeline, X, y, cv=cv,
                            scoring=["accuracy", "f1_weighted"])
```

---

## As métricas de avaliação

| Métrica | O que mede | Quando usar |
|---------|-----------|-------------|
| **Accuracy** (acurácia) | % de previsões corretas no total | Dados balanceados |
| **Precision** (precisão) | Dos que o modelo disse "positivo", quantos eram de verdade? | Quando falso positivo é caro |
| **Recall** (revocação) | Dos que eram positivos de verdade, quantos o modelo encontrou? | Quando falso negativo é caro (ex: doença) |
| **F1-Score** | Média harmônica entre Precision e Recall | Dados desbalanceados |
| **Matriz de Confusão** | Tabela de acertos e erros por classe | Sempre — dá o diagnóstico completo |

---

## As ferramentas do projeto

| Biblioteca | O que faz nesta aula |
|------------|---------------------|
| **scikit-learn** | `LogisticRegression`, `KNeighborsClassifier`, `Pipeline`, `ColumnTransformer`, `StandardScaler`, `SimpleImputer`, `cross_validate`, `StratifiedKFold` |
| **pandas** | Carrega e prepara os datasets, trata valores ausentes, separa features numéricas e categóricas |
| **seaborn** | Heatmap da matriz de confusão, pairplot para EDA dos datasets |
| **matplotlib** | Fronteira de decisão 2D, curvas de desempenho × K |

---

## Os datasets utilizados

| Dataset | Registros | Alvo | Tipo de classificação |
|---------|-----------|------|----------------------|
| Diabetes | 768 | Tem diabetes (0/1) | Binária |
| Palmer Penguins | 344 | Espécie (Adelie/Chinstrap/Gentoo) | Multi-classe (3 classes) |
| Titanic | 891 | Sobreviveu (0/1) | Binária |

---

## Os gráficos são salvos automaticamente

Toda vez que a aplicação é executada, os gráficos são exportados para `outputs/`:

- Formato: **PNG** (Portable Network Graphics)
- Resolução: **150 DPI** (adequada para impressão e apresentações)
- A pasta é criada automaticamente via `os.makedirs(exist_ok=True)`

---

## Autor

**Cláudio Ferreira Neves**
Especialista em Ciência de Dados e IA | Docente SENAI/SC

> Este material é de autoria exclusiva de **Cláudio Ferreira Neves**.
> É permitido o uso para fins de estudo, desde que citada a fonte.
> Reprodução total ou parcial sem crédito ao autor caracteriza **plágio**.
