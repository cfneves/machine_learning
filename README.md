# 🤖 Machine Learning — Portal de Aulas

<div align="center">

**Cláudio Ferreira Neves  ·  Especialista em Ciência de Dados e IA**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: CC BY](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)

</div>

> Material didático autoral. Reprodução ou uso sem crédito ao autor é considerado plágio.

---

Machine Learning é o conjunto de técnicas que permite que sistemas aprendam padrões a partir de dados históricos e façam previsões sobre situações novas, sem programar cada regra manualmente. Um modelo de crédito aprende com históricos de pagamento. Um filtro de spam aprende com e-mails rotulados. Um sistema de diagnóstico aprende com exames anteriores. Dados de entrada, padrão identificado, previsão na saída.

O que separa um projeto de ML que funciona de um que não funciona raramente é a escolha do algoritmo. Quase sempre é a qualidade dos dados. Dados inconsistentes ou mal estruturados geram modelos que erram com confiança, o que é pior do que não ter modelo nenhum. Antes de rodar qualquer algoritmo, é preciso entender o que os dados estão dizendo: quais variáveis importam, onde estão os ruídos, o que é sinal e o que é artefato da coleta.

Saber treinar um modelo é a parte mais fácil. O que diferencia um cientista de dados é saber fazer as perguntas certas antes de começar, e saber explicar os resultados para quem vai tomar a decisão.

---

## Sobre este material

Desenvolvido para quem começa do zero em Machine Learning. Cada aula tem teoria, código comentado e uma aplicação Streamlit que roda no navegador, sem abrir terminal.

A proposta não é decorar algoritmos. É entender o que acontece por baixo de cada técnica, por que ela existe, quando usá-la e como interpretar os resultados diante de um problema real.

---

## Como executar

```bash
# 1. Clone o repositório
git clone https://github.com/cfneves/machine_learning.git
cd machine_learning

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Inicie o portal
streamlit run app.py
```

Abra `http://localhost:8501` no navegador. O portal exibe todas as aulas e você navega entre elas pelos cards ou pelas abas no topo de cada página. A pasta `pages/` registra cada aula como uma página separada — nada de múltiplos terminais.

---

## Trilha de aprendizado

| Aula | Tema | Tópicos |
|------|------|---------|
| 01 | 🤖 Introdução ao ML | Regressão, Classificação, Train/Test Split, IRIS |
| 02 | 🔍 EDA | CRISP-DM, Univariada, Multivariada, IQR, Penguins |
| 03 | 📈 Regressão Linear e Polinomial | Linear Simples, Múltipla, Polinomial, Pipeline, SGD |
| 04 | 🔮 Regressão Logística e KNN | Sigmoid, KNN, GridSearchCV, Cross-Validation |
| 05 | ⚡ Naive Bayes e SVM | GaussianNB, SVM, Kernels, GridSearch, Wine |
| 06 | 🌳 Decision Tree e Random Forest | Gini, Bagging, Boosting, XGBoost, Feature Importance |
| 07 | 🎯 K-Means e PCA | Clusterização, Elbow, Silhouette, PCA, Mall Customers |
| 08 | 🧬 Comparação de Modelos | Comparativo cv=10, VotingClassifier, joblib, Alzheimer |

---

## Área do Aluno

O portal tem uma seção dedicada ao aluno acessível pelo botão "Área do Aluno" no Portal ou diretamente em `pages/Aluno.py`. Lá você encontra notebooks com exercícios em branco para praticar enquanto acompanha as aulas — células marcadas com ✍️ para preencher, com dicas no código.

| Aula | Notebook do aluno | Status |
|------|-------------------|--------|
| 01 | `Aula_01_Introducao_ao_ML_(aluno).ipynb` | ✅ Disponível |
| 02–08 | — | 🔒 Em breve |

---

## Estrutura do projeto

```
machine_learning/
├── app.py                        ← Apresentação do autor (página inicial)
├── nav.py                        ← Componente de abas — navegação entre aulas
├── requirements.txt              ← Dependências do projeto
├── README.md                     ← Este arquivo
├── pages/                        ← Multi-page Streamlit (cada arquivo = uma página)
│   ├── Portal.py                 ← Portal com todas as aulas
│   ├── Aluno.py                  ← Área do Aluno — download dos notebooks
│   ├── Aula_01.py ... Aula_08.py
└── aula_NN/                      ← Uma pasta por aula
    ├── app_streamlit.py          ← Aplicação interativa da aula
    ├── Aula_NN_Tema_(resolvido).ipynb  ← Notebook completo
    ├── Aula_NN_Tema_(aluno).ipynb      ← Notebook com exercícios (quando disponível)
    ├── requirements.txt
    └── README.md
```

---

## Tecnologias

| Biblioteca | Versão | Para que serve |
|------------|--------|----------------|
| `streamlit` | ≥ 1.36 | Interface web interativa |
| `pandas` | ≥ 2.0 | Manipulação de dados tabulares |
| `numpy` | ≥ 1.26 | Operações numéricas vetorizadas |
| `scikit-learn` | ≥ 1.4 | Algoritmos de Machine Learning |
| `xgboost` | ≥ 2.0 | Gradient boosting (aulas 06 e 08) |
| `joblib` | ≥ 1.3 | Serialização de modelos (aula 08) |
| `matplotlib` | ≥ 3.8 | Visualizações base |
| `seaborn` | ≥ 0.13 | Visualizações estatísticas |

---

## Autor

**Cláudio Ferreira Neves**
Especialista em Ciência de Dados e IA | Docente SENAI/SC

Uso para fins de estudo é permitido, desde que citada a fonte. Reprodução sem crédito ao autor caracteriza plágio.
