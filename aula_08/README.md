# 🧬 Aula 08 — Comparação de Modelos

<div align="center">

**Cláudio Ferreira Neves  ·  Especialista em Ciência de Dados e IA**

</div>

> Material didático autoral. Reprodução ou uso sem crédito ao autor é considerado plágio.

---

## O que é essa aula?

Esta é a aula final do curso, onde tudo se conecta.

Nas sete aulas anteriores você aprendeu algoritmos diferentes: Regressão Logística, KNN, Naive Bayes, SVM, Decision Tree, Random Forest e XGBoost. Em cada aula, você avaliou um modelo de forma isolada. Mas na prática, a pergunta real nunca é "esse modelo funciona?". É **"qual modelo funciona melhor para *este* problema?"**

Nesta aula você vai colocar todos os modelos para competir no mesmo dataset, em condições iguais, e tomar uma decisão baseada em evidências. Vai aprender também a **combinar os melhores modelos em um Ensemble** (conjunto), que muitas vezes supera qualquer modelo individual. E vai aprender a **salvar o modelo treinado em disco** para usar depois, o primeiro passo para um deploy real.

O dataset desta aula é o **Alzheimer's Disease** (diagnóstico da doença de Alzheimer), um problema médico real com mais de 35 features e um desafio genuíno de generalização.

---

## Como o projeto está organizado

```
aula_08/
│
├── app_streamlit.py                    # Aplicação web interativa — abre no navegador
├── README.md                           # Este arquivo
│
└── outputs/                            # Gráficos e modelos salvos automaticamente
```

> A pasta `outputs/` é criada automaticamente na primeira execução.

---

## Como rodar na sua máquina

### 1. Instale as dependências

```bash
pip install streamlit numpy pandas matplotlib seaborn scikit-learn xgboost joblib
```

> O XGBoost é opcional — se não estiver instalado, a seção correspondente fica desabilitada mas o restante funciona normalmente.

### 2. Abra a aplicação no navegador

```bash
streamlit run app_streamlit.py --server.port 8508
```

Acesse `http://localhost:8508` no navegador. Se estiver usando o portal principal, a navegação entre aulas já está configurada.

---

## O que você vai aprender — 4 seções

| # | Seção | O que é trabalhado |
|---|-------|--------------------|
| 1 | **EDA do Dataset Alzheimer** | Estrutura do dataset, distribuição do target, heatmap de correlação, pré-processamento com ColumnTransformer e exemplo de Pipeline |
| 2 | **Comparativo de Modelos (cv=10)** | Todos os algoritmos do curso avaliados com validação cruzada estratificada de 10 folds — tabela comparativa de Accuracy e F1 |
| 3 | **Voting Ensemble** | Como combinar modelos diferentes em um comitê de votação (hard e soft voting), construção do ensemble com os top 3 modelos, comparação com individuais |
| 4 | **Salvar e Carregar o Modelo com joblib** | Como serializar (salvar) um Pipeline treinado em disco, carregar depois e fazer previsões — o ciclo completo de um modelo em produção |

---

## O que dá pra mexer em tempo real

- **Seleção de modelos para comparar** → escolha quais algoritmos entram no ranking
- **Número de folds na validação cruzada** → ajuste de 5 a 15 e veja como a estabilidade das métricas muda
- **Tipo de votação no Ensemble** → compare Hard Voting (maioria de votos) vs Soft Voting (probabilidades médias)
- **Seleção dos top modelos para o Ensemble** → escolha quais 3 modelos formam o comitê
- **Demo de deploy** → treina, salva, carrega e faz uma previsão com o modelo salvo em disco

---

## Os conceitos principais

### Validação Cruzada Estratificada (cv=10)

Comparar modelos com um único split treino/teste é perigoso — o resultado pode depender de como os dados foram divididos. Com **cv=10**, cada modelo é avaliado 10 vezes em divisões diferentes, e o resultado final é a **média ± desvio padrão** dessas 10 avaliações. Isso dá muito mais confiança na comparação.

A versão **estratificada** garante que a proporção de cada classe seja mantida em cada fold. Isso importa especialmente em datasets com classes desbalanceadas como este.

```python
from sklearn.model_selection import cross_validate, StratifiedKFold

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for nome, modelo in modelos.items():
    resultado = cross_validate(modelo, X, y, cv=cv,
                               scoring=["accuracy", "f1_weighted"])
    print(f"{nome}: {resultado['test_accuracy'].mean():.3f}")
```

### Voting Ensemble — força no coletivo

Um **VotingClassifier** combina múltiplos modelos e decide pelo voto. Há duas formas de fazer isso:

- Hard Voting: cada modelo vota em uma classe e a classe com mais votos ganha.
- Soft Voting: cada modelo calcula a probabilidade de cada classe, as probabilidades são somadas e a classe com a maior soma ganha. Em geral produz resultados melhores, mas requer que todos os modelos tenham o método `predict_proba`.

A ideia por trás do Ensemble é que modelos diferentes erram em pontos diferentes. Quando você combina modelos com erros decorrelacionados, os erros individuais se cancelam e o conjunto fica mais forte do que qualquer membro isolado.

```python
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[
        ("rf",  pipeline_random_forest),
        ("svm", pipeline_svm),
        ("lr",  pipeline_logistica),
    ],
    voting="soft",   # usa probabilidades — geralmente melhor
)

ensemble.fit(X_train, y_train)
```

### joblib — salvando o modelo em disco

Depois de treinar e validar o melhor modelo, você precisa **salvá-lo** para usar no futuro, sem precisar retreinar. O `joblib` serializa o Pipeline inteiro (pré-processamento + modelo) em um único arquivo `.pkl`:

```python
import joblib

# Salvar o modelo treinado
joblib.dump(pipeline_treinado, "modelo_alzheimer.pkl")

# Carregar o modelo em outro momento (ou em outra máquina)
modelo = joblib.load("modelo_alzheimer.pkl")

# Fazer previsões diretamente — o pré-processamento já está embutido
previsao = modelo.predict(novos_dados)
```

Este é o fluxo de um modelo em produção: treina uma vez, salva, carrega onde precisar.

---

## O dataset utilizado

### Alzheimer's Disease Dataset

O dataset de Alzheimer contém dados clínicos e demográficos de pacientes, com o objetivo de classificar se o diagnóstico é positivo para a doença ou não.

| Informação | Valor |
|------------|-------|
| Fonte | Repositório público do GitHub (carregado via URL) |
| Features | Mais de 35 (numéricas e categóricas) |
| Alvo | Diagnóstico de Alzheimer (Positivo / Negativo) |
| Desafio | Features correlacionadas, valores ausentes, desbalanceamento |

É propositalmente mais complexo que os datasets das aulas anteriores, porque o objetivo desta aula final é simular um problema real, com toda a dificuldade que vem junto.

---

## O pipeline de um projeto real — CRISP-DM completo

Esta aula percorre as últimas fases do CRISP-DM:

| Fase | O que acontece nesta aula |
|------|--------------------------|
| Entendimento dos Dados | EDA completo do dataset Alzheimer |
| Preparação dos Dados | ColumnTransformer com imputação e escalonamento |
| Modelagem | Treinamento de todos os algoritmos do curso |
| Avaliação | Comparativo com cv=10, Ensemble |
| Implantação | Salvar e carregar o modelo com joblib |

---

## Resumo dos algoritmos do curso

| Aula | Algoritmo | Tipo | Quando usar |
|------|-----------|------|-------------|
| 03 | Regressão Linear/Polinomial | Supervisionado — Regressão | Prever valores contínuos |
| 04 | Regressão Logística | Supervisionado — Classificação | Classificação linear, interpretável |
| 04 | KNN | Supervisionado — Classificação | Dados localmente estruturados |
| 05 | Naive Bayes | Supervisionado — Classificação | Muitas features, dados textuais |
| 05 | SVM | Supervisionado — Classificação | Alta dimensão, margens claras |
| 06 | Decision Tree | Supervisionado — Classificação | Interpretabilidade é prioridade |
| 06 | Random Forest | Supervisionado — Classificação | Robustez e feature importance |
| 06 | XGBoost | Supervisionado — Classificação | Máxima performance |
| 07 | K-Means | Não Supervisionado — Clusterização | Segmentação sem rótulos |
| 07 | PCA | Não Supervisionado — Redução | Visualização, pré-processamento |

---

## Os gráficos são salvos automaticamente

Toda vez que a aplicação é executada, os gráficos são exportados para `outputs/`:

- Formato: **PNG** (Portable Network Graphics)
- Resolução: **150 DPI** (adequada para impressão e apresentações)
- A pasta é criada automaticamente via `os.makedirs(exist_ok=True)`

---

## Chegou até aqui

Se você acompanhou todas as 8 aulas, passou por toda a trilha de Machine Learning que este curso propõe: exploração e preparação de dados, algoritmos supervisionados e não supervisionados, avaliação honesta com validação cruzada e persistência de modelos com joblib. Agora o próximo passo é praticar com dados próprios.

---

## Autor

**Cláudio Ferreira Neves**
Especialista em Ciência de Dados e IA | Docente SENAI/SC

> Este material é de autoria exclusiva de **Cláudio Ferreira Neves**.
> É permitido o uso para fins de estudo, desde que citada a fonte.
> Reprodução total ou parcial sem crédito ao autor caracteriza **plágio**.
