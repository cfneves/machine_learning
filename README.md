# 🤖 Machine Learning — Portal de Aulas

<div align="center">

**Cláudio Ferreira Neves  ·  Especialista em Ciência de Dados e IA**

</div>

> Material didático autoral. Reprodução ou uso sem crédito ao autor é considerado plágio.

---

A inteligência artificial deixou de ser um diferencial para se tornar uma necessidade operacional. Empresas que ainda tomam decisões baseadas exclusivamente em intuição ou em relatórios históricos estão perdendo velocidade para concorrentes que já usam modelos preditivos no dia a dia — seja para antecipar demanda, detectar fraudes, personalizar experiências ou otimizar processos. Não é exagero dizer que a capacidade de extrair inteligência dos dados é hoje um dos ativos mais valiosos que uma organização pode desenvolver.

Mas dados brutos, por si só, não valem quase nada. O que transforma dado em vantagem competitiva é a qualidade com que ele foi coletado, organizado e interpretado. Dados inconsistentes, incompletos ou mal estruturados geram modelos que erram com confiança — e isso é pior do que não ter modelo nenhum. Por isso, antes de qualquer algoritmo entrar em cena, é preciso entender profundamente o que os dados estão dizendo: quais variáveis importam, onde estão os ruídos, quais padrões são reais e quais são artefatos do processo de coleta.

É nesse contexto que o profissional de dados se torna estratégico. Não apenas pela capacidade técnica de rodar um modelo, mas pela habilidade de fazer as perguntas certas — e de traduzir os resultados para quem toma decisão. O cientista de dados que sabe apenas codar é um executor. O que entende o negócio, questiona os dados e comunica os achados com clareza é o que realmente move o ponteiro.

Machine Learning é o conjunto de técnicas que permite que sistemas aprendam padrões a partir de dados históricos e façam previsões ou classificações sobre situações novas — sem que cada regra precise ser programada manualmente. Da detecção de anomalias em tempo real à recomendação de produtos, da análise de risco de crédito ao diagnóstico médico assistido, as aplicações são vastas. Dominar esses fundamentos não é opcional para quem trabalha com dados: é o que separa análises descritivas do passado de decisões inteligentes sobre o futuro.

---

## Sobre este material

Este é um material didático autoral desenvolvido por **Cláudio Ferreira Neves**, com foco no aprendizado prático de Machine Learning do zero. Cada aula combina teoria direta com exemplos reais, código comentado e aplicações interativas construídas em Streamlit — acessíveis pelo navegador, sem precisar rodar nada no terminal.

A proposta não é decorar algoritmos. É entender o que acontece por baixo de cada técnica, por que ela existe, quando usá-la e como interpretar os resultados no contexto de um problema real.

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
