# 🤖 Machine Learning — Portal de Aulas

<div align="center">

**Cláudio Ferreira Neves  ·  Especialista em Ciência de Dados e IA**

</div>

> Material didático autoral. Reprodução ou uso sem crédito ao autor é considerado plágio.

---

Machine Learning é o conjunto de técnicas que permite que sistemas aprendam padrões a partir de dados históricos e façam previsões sobre situações novas, sem que cada regra seja programada manualmente. Um modelo de crédito aprende com históricos de pagamento. Um filtro de spam aprende com e-mails rotulados. Um sistema de diagnóstico aprende com exames anteriores. A lógica é sempre a mesma: dados de entrada, padrão identificado, previsão na saída.

O que separa um projeto de ML que funciona de um que não funciona raramente é a escolha do algoritmo. Quase sempre é a qualidade dos dados. Dados inconsistentes ou mal estruturados geram modelos que erram com confiança, o que é pior do que não ter modelo nenhum. Antes de rodar qualquer algoritmo, é preciso entender o que os dados estão dizendo: quais variáveis importam, onde estão os ruídos, o que é sinal e o que é artefato da coleta.

Saber treinar um modelo é a parte mais fácil. O que realmente diferencia o trabalho de um cientista de dados é saber fazer as perguntas certas antes de começar, e saber explicar os resultados para quem vai tomar a decisão.

---

## Sobre este material

Este é um material didático de **Cláudio Ferreira Neves**, desenvolvido para quem está começando do zero em Machine Learning. Cada aula tem teoria, código comentado e uma aplicação Streamlit que roda no navegador, sem precisar abrir terminal.

A proposta não é decorar algoritmos. É entender o que acontece por baixo de cada técnica, por que ela existe, quando usá-la e como interpretar os resultados diante de um problema real.

---

## Estrutura do projeto

```
machine_learning/
├── app.py                        ← Portal de navegação (página inicial)
├── nav.py                        ← Componente de abas — navegação entre aulas
├── README.md                     ← Este arquivo
├── pages/                        ← Multi-page Streamlit (cada arquivo = uma página)
│   ├── Aula_01.py
│   ├── Aula_02.py
│   ├── Aula_03.py
│   ├── Aula_04.py
│   ├── Aula_05.py
│   ├── Aula_06.py
│   ├── Aula_07.py
│   └── Aula_08.py
├── aula_01/                      ← Introdução ao Machine Learning
│   ├── app_streamlit.py
│   ├── requirements.txt
│   └── README.md
├── aula_02/                      ← Análise Exploratória de Dados (EDA)
│   ├── app_streamlit.py
│   ├── requirements.txt
│   └── README.md
├── aula_03/                      ← Regressão Linear e Polinomial
│   ├── app_streamlit.py
│   ├── requirements.txt
│   └── README.md
├── aula_04/                      ← Regressão Logística e KNN
│   ├── app_streamlit.py
│   ├── requirements.txt
│   └── README.md
├── aula_05/                      ← Naive Bayes e SVM
│   ├── app_streamlit.py
│   ├── requirements.txt
│   └── README.md
├── aula_06/                      ← Decision Tree e Random Forest
│   ├── app_streamlit.py
│   ├── requirements.txt
│   └── README.md
├── aula_07/                      ← K-Means e PCA
│   ├── app_streamlit.py
│   ├── requirements.txt
│   └── README.md
└── aula_08/                      ← Comparação de Modelos
    ├── app_streamlit.py
    └── README.md
```

---

## Como executar

Um único comando inicia todo o projeto. Um terminal, um processo, todas as aulas.

```bash
streamlit run app.py
```

> Abra `http://localhost:8501` no navegador. O portal exibe todas as aulas e você navega entre elas clicando nos cards ou nas abas no topo de cada página.

**Não é necessário abrir múltiplos terminais.** O projeto usa a arquitetura multi-page nativa do Streamlit: a pasta `pages/` registra cada aula como uma página separada, e o `nav.py` renderiza a barra de abas centralizada em todas elas.

---

## Trilha de aprendizado

| Aula | Tema | Status | Tópicos |
|------|------|--------|---------|
| 01 | Introdução ao ML | ✅ Disponível | Regressão, Classificação, Train/Test Split, IRIS |
| 02 | EDA | ✅ Disponível | CRISP-DM, Univariada, Multivariada, IQR, Penguins |
| 03 | Regressão Linear e Polinomial | ✅ Disponível | Linear Simples, Múltipla, Polinomial, Pipeline, SGD |
| 04 | Regressão Logística e KNN | ✅ Disponível | Sigmoid, KNN, GridSearchCV, Cross-Validation |
| 05 | Naive Bayes e SVM | ✅ Disponível | GaussianNB, SVM, Kernels, GridSearch, Wine |
| 06 | Decision Tree e Random Forest | ✅ Disponível | Gini, Bagging, Boosting, XGBoost, Feature Importance |
| 07 | K-Means e PCA | ✅ Disponível | Clusterização, Elbow, Silhouette, PCA, Mall Customers |
| 08 | Comparação de Modelos | ✅ Disponível | Comparativo cv=10, VotingClassifier, joblib, Alzheimer |

---

## Tecnologias

| Biblioteca | Versão recomendada | Para que serve |
|------------|--------------------|----------------|
| `streamlit` | ≥ 1.32 | Interface web interativa |
| `pandas` | ≥ 2.0 | Manipulação de dados tabulares |
| `numpy` | ≥ 1.26 | Operações numéricas vetorizadas |
| `scikit-learn` | ≥ 1.4 | Algoritmos de Machine Learning |
| `xgboost` | ≥ 2.0 | Gradient boosting de alta performance (aulas 06 e 08) |
| `joblib` | ≥ 1.3 | Serialização de modelos treinados (aula 08) |
| `matplotlib` | ≥ 3.8 | Visualizações base |
| `seaborn` | ≥ 0.13 | Visualizações estatísticas |

---

## Autor

**Cláudio Ferreira Neves**
Especialista em Ciência de Dados e IA | Docente SENAI/SC

> Este material é de autoria exclusiva de **Cláudio Ferreira Neves**.
> É permitido o uso para fins de estudo, desde que citada a fonte.
> Reprodução total ou parcial sem crédito ao autor caracteriza **plágio**.
