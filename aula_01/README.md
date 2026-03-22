# 🤖 Introdução ao Machine Learning (Aprendizado de Máquina)

<div align="center">

**Cláudio Ferreira Neves  ·  Especialista em Ciência de Dados e IA**

</div>

> Material didático autoral. Reprodução ou uso sem crédito ao autor é considerado plágio.

---

## 📋 O que é esse projeto?

Se você está começando no mundo de Machine Learning (Aprendizado de Máquina) e se sente perdido com tantos termos e conceitos novos — relaxa, esse projeto foi feito pensando exatamente em você.

A ideia aqui é simples: **aprender fazendo**. Cada seção mostra o conceito, explica o código linha por linha e deixa você interagir com os resultados em tempo real. Nada de decorar fórmula. O foco é entender o que está acontecendo e por quê.

O projeto tem dois arquivos principais:

- **`01_ml_intro.py`** — o script (roteiro) Python original, onde tudo começou. É aqui que estão os algoritmos (procedimentos/fórmulas), os gráficos e a lógica de Machine Learning. Ideal para rodar direto no terminal e ver os resultados.
- **`app_streamlit.py`** — foi construído **a partir** do script acima. Ele pega todo esse conteúdo e transforma em uma aplicação web interativa, com explicações, controles e visualizações. É o que você acessa pelo navegador.

---

## 🌐 Acesso Online

> Em breve — o link será adicionado assim que a aplicação for publicada (deploy):
>
> **[Link da aplicação — em breve]**

---

## 📁 Como o projeto está organizado

```
ml-introducao/
│
├── app_streamlit.py     # A aplicação web interativa — é aqui que tudo acontece no navegador
├── 01_ml_intro.py       # O script Python de origem — roda no terminal, sem interface
├── requirements.txt     # Lista de bibliotecas necessárias para o projeto funcionar
├── README.md            # Este arquivo — a porta de entrada do projeto
│
└── outputs/             # Pasta onde os gráficos são salvos automaticamente (150 DPI)
    ├── 01_scatter_sem_classes.png
    ├── 02_scatter_com_classes.png
    ├── 03_regressao_linear_manual.png
    ├── 04_regressao_linear_comparacao.png
    ├── 05_regressao_linear_previsao.png
    ├── 06_regressao_polinomial_grauN.png
    ├── 07_poly_underfitting_overfitting.png
    ├── 08_classificacao_dataset.png
    ├── 09_classificacao_fronteira.png
    ├── 10_classificacao_novo_ponto.png
    ├── 11_divisao_treino_teste.png
    ├── 12_treino_vs_accuracy.png
    ├── 13_iris_sepala.png
    ├── 14_iris_petala.png
    ├── 15_iris_confusion_matrix.png
    └── 16_iris_fronteira_decisao.png
```

> A pasta `outputs/` é criada automaticamente quando você roda a aplicação — não precisa criar na mão.

---

## 🚀 Como rodar na sua máquina

### 1. Instale as dependências (bibliotecas necessárias)

Abra o terminal na pasta do projeto e rode:

```bash
pip install -r requirements.txt
```

Se preferir instalar uma por uma:

```bash
pip install streamlit numpy matplotlib seaborn scikit-learn pandas
```

### 2. Abrir a aplicação no navegador

```bash
streamlit run app_streamlit.py
```

O próprio Streamlit abre o navegador automaticamente em `http://localhost:8501`. Se não abrir, copie esse endereço e cole na barra do navegador.

### 3. Rodar o script Python puro (sem interface)

```bash
python 01_ml_intro.py
```

Os gráficos aparecem em janelas do Matplotlib (uma por vez). Feche uma para avançar para a próxima.

---

## 📚 O que você vai aprender — 6 seções

| # | Seção | O que é trabalhado |
|---|-------|--------------------|
| 1 | 📊 **Representação de Dados** | Como os dados são organizados em arrays (vetores/matrizes), indexação e visualização com scatter plots (gráficos de dispersão) |
| 2 | 📈 **Regressão Linear Simples** | Como encontrar a melhor reta para um conjunto de dados — na mão e com scikit-learn |
| 3 | 🔢 **Regressão Polinomial** | Quando a reta não é suficiente — curvas, underfitting (subajuste) e overfitting (sobreajuste) |
| 4 | 🎯 **Classificação Binária** | Como o modelo aprende a separar duas classes e desenha uma fronteira de decisão (decision boundary) |
| 5 | ✂️ **Divisão Treino / Teste** | Por que nunca avaliamos o modelo com os mesmos dados em que ele treinou |
| 6 | 🌸 **Dataset IRIS — Pipeline Completo** | Análise exploratória (EDA) → treino → avaliação com acurácia (accuracy), relatório de classificação e matriz de confusão |

---

## 🎛️ O que dá pra mexer em tempo real

Uma das partes mais legais da aplicação é poder ajustar os parâmetros e ver o modelo reagindo na hora. Aqui está o que você pode explorar:

- **Número de amostras** e **semente aleatória** (random state) → controla quantos dados são gerados e garante que os resultados sejam sempre iguais quando você repetir
- **Nível de ruído** → simula dados mais "bagunçados" ou mais organizados
- **β₀ e β₁** → os coeficientes reais da reta — veja se o modelo consegue descobrir esses valores
- **Grau do polinômio** → explore como um modelo pode ficar simples demais (underfitting) ou complexo demais (overfitting)
- **Separação entre classes** → quanto mais próximas as classes, mais difícil para o modelo
- **Proporção treino/teste** → veja como a quantidade de dados afeta as métricas (indicadores de desempenho)
- **Previsão de novo ponto** → informe um ponto e veja o que o modelo classifica
- **Seleção de features** (características) → compare combinações de variáveis no dataset (conjunto de dados) IRIS

---

## 🧰 As ferramentas do projeto — quem são, para que servem e por que foram escolhidas

Antes de usar uma ferramenta, é importante entender o que ela faz. Essa seção é especialmente para quem está começando: conhecer o papel de cada biblioteca vai te ajudar a tomar decisões melhores nos seus próprios projetos.

---

### 🔢 NumPy — a base de tudo que envolve números

Sabe quando você precisa trabalhar com uma lista de 10.000 números e fazer operações matemáticas em todos eles ao mesmo tempo? É exatamente isso que o NumPy (Numerical Python — Python Numérico) resolve, de forma rápida e eficiente.

Ele introduz o conceito de **array** (vetor ou matriz de números), que é completamente diferente de uma lista comum do Python. Uma lista é lenta para cálculos matemáticos. Um array NumPy é otimizado para isso.

**O que ele faz por nós aqui:**
- Cria e organiza os dados de entrada (features — características) e saída (labels — rótulos)
- Gera números aleatórios com controle de semente (para resultados reproduzíveis — repetíveis)
- Calcula médias, somas e outras operações estatísticas com uma linha de código

**Por que é a escolha certa:** scikit-learn, TensorFlow e PyTorch foram todos construídos sobre o NumPy. Aprender NumPy é aprender o idioma base da área. Operações vetorizadas (aplicadas a todos os elementos de uma vez) chegam a ser 100× mais rápidas do que um `for` comum.

```python
import numpy as np

# Gera 100 valores aleatórios entre 0 e 2 — shape (100, 1): 100 linhas, 1 coluna
X = 2 * np.random.rand(100, 1)

# Calcula a média de todos os valores — uma linha, resultado imediato
media = np.mean(X)
```

---

### 📊 Matplotlib — transformando dados em gráficos

Dados soltos em uma tabela dizem pouco. Um gráfico bem feito conta a história por inteiro. O Matplotlib é a biblioteca mais tradicional para criar gráficos em Python — existe desde 2003 e ainda hoje é o ponto de partida de qualquer visualização.

O nome vem de "MATLAB-like plotting library" (biblioteca de gráficos similar ao MATLAB), e a proposta é exatamente essa: controle total sobre cada detalhe visual do gráfico.

**O que ele faz por nós aqui:**
- Cria scatter plots (gráficos de dispersão) para ver como os dados estão distribuídos
- Desenha a reta (ou curva) do modelo sobre os dados reais
- Compara visualmente diferentes modelos no mesmo gráfico
- Salva os gráficos em alta resolução na pasta `outputs/`

**Por que é a escolha certa:** Seaborn e Plotly são construídos sobre ele. Entender Matplotlib significa entender como qualquer gráfico Python funciona por baixo.

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))           # tamanho da figura em polegadas
plt.scatter(X, y, color="blue",      # pontos azuis = dados reais
            label="Dados reais")
plt.plot(X, y_pred, color="red",     # linha vermelha = previsão do modelo
         label="Modelo")
plt.legend()                          # exibe a legenda no gráfico
plt.show()                            # renderiza (mostra) o gráfico na tela
```

---

### 🎨 Seaborn — gráficos estatísticos com menos código

Se o Matplotlib é a fundação, o Seaborn é o acabamento. Ele foi construído **sobre** o Matplotlib e é especializado em gráficos voltados para análise estatística. Com muito menos linhas de código, você chega em visualizações mais elegantes e informativas.

**O que ele faz por nós aqui:**
Basicamente uma coisa — mas fundamental: a **matriz de confusão** (confusion matrix). Esse gráfico mostra exatamente onde o modelo acertou e onde errou, classe por classe. Com Seaborn, criamos isso em 3 linhas com anotações automáticas e escala de cores — o que levaria dezenas de linhas no Matplotlib puro.

**Por que é a escolha certa:** Para análise estatística e avaliação de modelos, Seaborn é a escolha mais eficiente. Menos código, resultado mais legível, integração nativa com tabelas de dados (DataFrames) do Pandas.

```python
import seaborn as sns

# Heatmap (mapa de calor) da matriz de confusão
# annot=True: mostra os números dentro de cada célula
# cmap="Blues": quanto mais escuro, maior o valor
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Setosa", "Versicolor", "Virginica"],
            yticklabels=["Setosa", "Versicolor", "Virginica"])
plt.xlabel("Previsto")   # eixo X: o que o modelo achou
plt.ylabel("Real")       # eixo Y: o valor verdadeiro (gabarito)
```

---

### 🤖 scikit-learn — a biblioteca de Machine Learning

Se você vai trabalhar com Machine Learning em Python, o scikit-learn (também chamado de sklearn) vai estar em praticamente todos os seus projetos. Reúne dezenas de algoritmos, ferramentas de pré-processamento e métricas de avaliação com uma interface consistente.

**O que ele faz por nós aqui:**
- Treina modelos de regressão (`LinearRegression`) e classificação (`LogisticRegression`)
- Cria features (características) polinomiais com `PolynomialFeatures`
- Divide os dados em treino e teste com `train_test_split` (divisão treino/teste)
- Avalia os modelos: acurácia (accuracy), RMSE, R², matriz de confusão, relatório de classificação
- Encadeia etapas com `Pipeline` (sequência de transformações + modelo)

**Por que é a escolha certa:** A interface é consistente em todos os algoritmos: `fit()` treina, `predict()` prevê, `score()` avalia. Aprenda com um e você já sabe usar os outros.

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Nunca avalie o modelo com os mesmos dados do treino!
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)      # fit = treinar o modelo
y_pred = model.predict(X_test)   # predict = fazer as previsões
r2 = model.score(X_test, y_test) # score = avaliar o desempenho
```

---

### 🌐 Streamlit — de script Python para aplicação web

Você construiu um modelo de Machine Learning e quer mostrar os resultados para alguém que não sabe Python. O Streamlit resolve isso. É um framework Python que transforma qualquer script em uma aplicação web, sem precisar escrever HTML, CSS ou JavaScript.

**O que ele faz por nós aqui:**
- Cria a interface completa de navegação entre as seções
- Adiciona sliders (controles deslizantes), botões e seletores interativos
- Exibe os gráficos do Matplotlib diretamente na página
- Permite publicar (fazer deploy) a aplicação online de graça

**Por que é a escolha certa:** Em vez de enviar um notebook que poucos sabem abrir, você entrega uma aplicação que qualquer pessoa acessa pelo navegador.

```python
import streamlit as st

st.title("Meu Primeiro Modelo de ML")

# Slider (controle deslizante): min=10, max=500, padrão=100
amostras = st.slider("Quantas amostras você quer?", 10, 500, 100)

st.write(f"Você escolheu {amostras} amostras. O gráfico vai atualizar automaticamente!")

# O gráfico aparece na página web — sem abrir janela separada
fig, ax = plt.subplots()
ax.scatter(range(amostras), range(amostras))
st.pyplot(fig)
```

---

## 📐 Os algoritmos — entenda a matemática por trás

Não se preocupe se as fórmulas parecerem intimidadoras no começo. O que importa é entender a ideia geral de cada uma.

### Regressão Linear Simples (Simple Linear Regression)

A ideia é encontrar a reta que melhor passa pelos dados. "Melhor" aqui significa: a reta que deixa os erros (distância entre o ponto real e a reta) o menor possível. O método usado se chama OLS (Ordinary Least Squares — Mínimos Quadrados Ordinários):

$$\hat{y} = \beta_0 + \beta_1 X \qquad \beta_1 = \frac{\sum(X_i - \bar{X})(y_i - \bar{y})}{\sum(X_i - \bar{X})^2}$$

### Regressão Polinomial (Polynomial Regression)

Às vezes os dados não seguem uma reta — eles formam uma curva. A regressão polinomial resolve isso adicionando potências de X como novas variáveis:

$$\hat{y} = \beta_0 + \beta_1 X + \beta_2 X^2 + \cdots + \beta_d X^d$$

### Regressão Logística (Logistic Regression — usada para Classificação)

Apesar do nome conter "regressão", ela é usada para classificação. Em vez de prever um valor contínuo, ela calcula a **probabilidade** (chance) de um ponto pertencer a uma determinada classe — usando a função sigmoide (função em forma de S):

$$P(y=1 \mid X) = \frac{1}{1 + e^{-(\beta_0 + \boldsymbol{\beta}^T X)}}$$

---

## 📸 Os gráficos são salvos automaticamente

Toda vez que você roda a aplicação, os gráficos são exportados para a pasta `outputs/`:

- Formato: **PNG** (Portable Network Graphics — imagem sem perda de qualidade)
- Resolução: **150 DPI** (dots per inch — pontos por polegada — qualidade suficiente para impressão)
- Nomeados em sequência e com nomes descritivos, fáceis de identificar

---

## 👨‍💻 Autor

**Cláudio Ferreira Neves**
Especialista em Ciência de Dados e IA

> Este material é de autoria exclusiva de **Cláudio Ferreira Neves**.
> É permitido o uso para fins de estudo, desde que citada a fonte.
> Reprodução total ou parcial sem crédito ao autor caracteriza **plágio**.
