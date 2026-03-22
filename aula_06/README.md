# 🌳 Aula 06 — Decision Tree e Random Forest

<div align="center">

**Cláudio Ferreira Neves  ·  Especialista em Ciência de Dados e IA**

</div>

> Material didático autoral. Reprodução ou uso sem crédito ao autor é considerado plágio.

---

## O que é essa aula?

Se você já tomou uma decisão seguindo uma sequência de perguntas, "o paciente tem mais de 50 anos? Sim, faz o exame. Não, observa primeiro", você já pensou como uma **árvore de decisão**. O algoritmo divide os dados em grupos menores até chegar a uma conclusão.

O problema das árvores individuais é que tendem a memorizar o treino em vez de aprender padrões gerais. Criar centenas delas e deixar que votem juntas resolve boa parte disso. Esse é o princípio da **Random Forest**. O **XGBoost** vai além: treina as árvores em sequência, cada uma focando nos erros da anterior.

Esta aula usa o dataset **Breast Cancer**, diagnóstico de câncer de mama, onde a precisão do modelo tem consequências reais.

---

## Como o projeto está organizado

```
aula_06/
│
├── app_streamlit.py                            # Aplicação web interativa — abre no navegador
├── aula06_decision_tree_random_forest.py       # Script Python principal — roda no terminal
├── requirements.txt                            # Bibliotecas necessárias
├── README.md                                   # Este arquivo
│
└── outputs/                                    # Gráficos salvos automaticamente (150 DPI)
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
pip install streamlit numpy pandas matplotlib seaborn scikit-learn xgboost
```

> O XGBoost é opcional — se não estiver instalado, a seção correspondente fica desabilitada mas o restante da aula funciona normalmente.

### 2. Abra a aplicação no navegador

```bash
streamlit run app_streamlit.py --server.port 8506
```

Acesse `http://localhost:8506` no navegador. Se estiver usando o portal principal, a navegação entre aulas já está configurada.

---

## O que você vai aprender — 4 seções

| # | Seção | O que é trabalhado |
|---|-------|--------------------|
| 1 | **Decision Tree** | Critérios de divisão (Gini e Entropia), controle de profundidade para evitar overfitting, visualização da árvore, Feature Importance e validação cruzada cv=10 |
| 2 | **Random Forest** | Como funciona o Bagging, efeito do número de árvores (n_estimators), Feature Importance comparativo, curva de desempenho |
| 3 | **XGBoost** | Bagging vs Boosting, como o XGBoost corrige erros iterativamente, Feature Importance e validação cruzada |
| 4 | **Comparativo Final** | Decision Tree vs Random Forest vs XGBoost no Breast Cancer — lado a lado |

---

## O que dá pra mexer em tempo real

- **Profundidade máxima da árvore (max_depth)** → veja o overfitting acontecer em tempo real quando a árvore cresce demais
- **Critério de divisão** → compare Gini e Entropia e entenda quando cada um faz diferença
- **Número de árvores (n_estimators)** → acompanhe como a acurácia melhora conforme a floresta cresce — e quando para de melhorar
- **Curva de overfitting** → gráfico interativo que mostra a diferença entre desempenho no treino e no teste conforme max_depth aumenta
- **Visualização da árvore** → veja a estrutura real de uma árvore com max_depth=3

---

## Os conceitos principais

### Decision Tree — aprendendo com perguntas

A árvore funciona dividindo os dados com base na feature que melhor separa as classes. "Melhor separar" pode ser medido de duas formas:

**Gini Impurity** (impureza de Gini) — mede a probabilidade de classificar errado um elemento escolhido aleatoriamente:

$$\text{Gini}(p) = 1 - \sum_{k} p_k^2$$

**Entropia** (Information Gain — Ganho de Informação) — mede a desordem no nó:

$$H(p) = -\sum_{k} p_k \log_2(p_k)$$

Na prática, as duas métricas levam a resultados parecidos. Gini é ligeiramente mais rápida.

Sem limite de profundidade, a árvore divide até ter um nó por amostra: decora tudo e não generaliza nada. O `max_depth` é o principal parâmetro de controle.

### Random Forest — força no coletivo

A Random Forest resolve o overfitting das árvores individuais com duas técnicas:

1. **Bagging (Bootstrap Aggregating):** cada árvore treina com uma amostra *com reposição* do dataset — os dados são ligeiramente diferentes para cada árvore
2. **Aleatoriedade de features:** em cada divisão, apenas um subconjunto aleatório de features é considerado — isso garante diversidade entre as árvores

O resultado final é a **votação** de todas as árvores. Erros individuais se cancelam, e o modelo final é muito mais estável.

### XGBoost — árvores que se corrigem

O Random Forest treina todas as árvores em paralelo e de forma independente. O **XGBoost** segue outra lógica, o **Boosting**: treina uma árvore, analisa onde errou, e a próxima árvore foca especificamente nesses erros. Iteração após iteração, o modelo vai ficando mais preciso.

Domina competições de ML em dados tabulares.

### Feature Importance — quais variáveis importam?

O Decision Tree, o Random Forest e o XGBoost calculam a importância de cada feature: quanto cada variável contribuiu para as divisões ao longo de todas as árvores. Modelos como o SVM não oferecem essa informação diretamente, o que torna a interpretabilidade uma vantagem prática desses algoritmos baseados em árvore.

```python
importancias = modelo.feature_importances_
# Retorna um array com a contribuição de cada feature (soma = 1)
```

---

## Bagging vs Boosting — qual a diferença?

| Característica | Bagging (Random Forest) | Boosting (XGBoost) |
|---------------|------------------------|---------------------|
| Treinamento | Paralelo | Sequencial |
| Cada árvore foca em | Amostra aleatória | Erros da árvore anterior |
| Resultado final | Votação simples | Soma ponderada |
| Sensibilidade a ruído | Baixa | Pode ser maior |
| Velocidade de treino | Rápida (paralelizável) | Mais lenta |

---

## O dataset utilizado

### Breast Cancer Wisconsin

O **Breast Cancer dataset** do scikit-learn contém 569 amostras de diagnóstico de câncer de mama, com 30 features calculadas a partir de imagens de biópsia (raio, textura, perímetro, área, suavidade, etc.).

| Informação | Valor |
|------------|-------|
| Amostras | 569 |
| Features | 30 |
| Classes | 2 (Maligno / Benigno) |
| Distribuição | ~63% Benigno, ~37% Maligno |

É um dataset realista e ligeiramente desbalanceado — por isso a validação cruzada estratificada é usada.

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
