# 📈 Aula 03 — Regressão Linear e Polinomial

<div align="center">

**Cláudio Ferreira Neves  ·  Especialista em Ciência de Dados e IA**

</div>

> Material didático autoral. Reprodução ou uso sem crédito ao autor é considerado plágio.

---

## O que é essa aula?

Quando a resposta que você quer prever é um número, o problema é de **regressão**. Preço de imóvel, custo de seguro, temperatura de amanhã. É o ponto de partida de Machine Learning e aparece o tempo todo na prática.

Esta aula começa com o método mais simples, usando uma única variável para prever, e vai até pipelines que combinam transformações e modelo em um fluxo reproduzível.

O projeto usa três datasets reais: um dataset artificial de imóveis, o **California Housing** (preços de casas na Califórnia) e o **Insurance** (custo de plano de saúde). Os problemas são reais.

---

## Como o projeto está organizado

```
aula_03/
│
├── app_streamlit.py                          # Aplicação web interativa — abre no navegador
├── aula03_regressao_linear_polinomial.py     # Script Python principal — roda no terminal
├── requirements.txt                          # Bibliotecas necessárias
├── README.md                                 # Este arquivo
│
└── outputs/                                  # Gráficos salvos automaticamente (150 DPI)
```

> A pasta `outputs/` é criada automaticamente na primeira execução — não precisa criar na mão.

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
streamlit run app_streamlit.py --server.port 8503
```

Acesse `http://localhost:8503` no navegador. Se estiver usando o portal principal, a navegação entre aulas já está configurada automaticamente.

### 3. Rodar o script Python puro (sem interface)

```bash
python aula03_regressao_linear_polinomial.py
```

Os resultados aparecem no terminal e os gráficos são salvos na pasta `outputs/`.

---

## O que você vai aprender — 5 seções

| # | Seção | O que é trabalhado |
|---|-------|--------------------|
| 1 | **Regressão Linear Simples** | Como encontrar a melhor reta para um conjunto de dados — parâmetros ajustáveis, métricas R² e RMSE |
| 2 | **Regressão Linear Múltipla** | Usar mais de uma variável para prever — datasets de imóveis, Insurance e California Housing com Pipeline |
| 3 | **Gradiente Descendente** | Como o modelo aprende iterativamente — visualização da função de custo e `SGDRegressor` |
| 4 | **Regressão Polinomial** | Quando a reta não dá conta — curvas, underfitting, overfitting e `PolynomialFeatures` |
| 5 | **Pipeline Completo** | Combinando `StandardScaler` + `LinearRegression` ou `PolynomialFeatures` em um fluxo reproduzível |

---

## O que dá pra mexer em tempo real

A aplicação interativa deixa você experimentar sem escrever código. Aqui estão os controles principais:

- **Nível de ruído** → adicione mais ou menos bagunça aos dados e veja como o modelo reage
- **Proporção treino/teste** → veja como a quantidade de dados de treino afeta o desempenho
- **Grau do polinômio** → explore o ponto onde o modelo começa a decorar os dados em vez de aprender (overfitting)
- **Taxa de aprendizado e épocas** → controle como o gradiente descendente converge
- **Seleção de features** → escolha quais variáveis incluir no modelo múltiplo e compare os resultados

---

## Os conceitos principais — entenda antes de rodar

### Regressão Linear Simples

O objetivo é encontrar a reta que passa mais perto de todos os pontos ao mesmo tempo. "Mais perto" significa minimizar a soma dos quadrados das distâncias entre os pontos reais e a reta. Esse método se chama OLS (Ordinary Least Squares, ou Mínimos Quadrados Ordinários):

$$\hat{y} = \beta_0 + \beta_1 X$$

Onde **β₀** é onde a reta cruza o eixo Y (intercepto) e **β₁** é a inclinação da reta.

### Regressão Linear Múltipla

A mesma ideia, mas agora com várias variáveis de entrada. Cada variável tem seu próprio coeficiente:

$$\hat{y} = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_p X_p$$

### Regressão Polinomial

Quando os dados seguem uma curva em vez de uma reta, adicionamos potências de X como novas colunas — e o modelo ainda é linear nos coeficientes, só os dados que mudaram de forma:

$$\hat{y} = \beta_0 + \beta_1 X + \beta_2 X^2 + \cdots + \beta_d X^d$$

**Atenção:** quanto maior o grau, mais o modelo se adapta ao treino — mas pode perder a capacidade de generalizar para dados novos. Isso se chama **overfitting** (sobreajuste).

### Gradiente Descendente

Em vez de calcular os coeficientes numa fórmula fechada, o gradiente descendente os ajusta aos poucos: começa com valores aleatórios, calcula o erro, e dá um pequeno passo na direção que reduz esse erro. A **taxa de aprendizado** controla o tamanho desse passo.

$$\theta := \theta - \alpha \cdot \nabla_\theta J(\theta)$$

---

## As métricas de avaliação

| Métrica | O que mede | Como interpretar |
|---------|-----------|-----------------|
| **R²** (coeficiente de determinação) | Proporção da variância explicada pelo modelo | Vai de 0 a 1 — quanto mais perto de 1, melhor |
| **RMSE** (Root Mean Squared Error) | Erro médio nas mesmas unidades do target | Quanto menor, melhor — penaliza erros grandes |
| **MAE** (Mean Absolute Error) | Erro absoluto médio | Mais intuitivo — a distância média entre previsto e real |

---

## As ferramentas do projeto

### scikit-learn — o coração do projeto

Aqui estão as principais classes usadas e o que cada uma faz:

- `LinearRegression` — ajusta uma reta (ou hiperplano) aos dados usando OLS
- `SGDRegressor` — versão do gradiente descendente estocástico (atualiza os pesos uma amostra por vez)
- `PolynomialFeatures` — cria novas colunas com as potências das features originais (X², X³...)
- `StandardScaler` — coloca todas as variáveis na mesma escala (média 0, desvio padrão 1). Necessário antes do gradiente descendente
- `Pipeline` — encadeia transformações e modelo em sequência, garantindo que nenhuma informação do teste vaze para o treino (evita data leakage)
- `ColumnTransformer` — aplica transformações diferentes para colunas numéricas e categóricas ao mesmo tempo

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("poly",   PolynomialFeatures(degree=2)),
    ("model",  LinearRegression()),
])

pipe.fit(X_train, y_train)   # treina tudo de uma vez
y_pred = pipe.predict(X_test) # transforma e prevê em sequência
```

---

## Os datasets utilizados

| Dataset | Registros | Variável alvo | Objetivo |
|---------|-----------|---------------|----------|
| Imóveis (artificial) | Gerado | Preço (R$) | Demonstrar regressão simples e múltipla de forma intuitiva |
| Insurance | 1.338 | Charges (USD) | Prever custo de seguro de saúde com variáveis reais |
| California Housing | 20.640 | Valor médio das casas | Regressão múltipla em escala real com 8 features |

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
