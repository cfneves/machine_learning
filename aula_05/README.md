# 🧠 Aula 05 — Naive Bayes e SVM

<div align="center">

**Cláudio Ferreira Neves  ·  Especialista em Ciência de Dados e IA**

</div>

> Material didático autoral. Reprodução ou uso sem crédito ao autor é considerado plágio.

---

## O que é essa aula?

Na aula anterior você aprendeu a classificar com Regressão Logística e KNN. Agora vamos explorar duas abordagens completamente diferentes, e cada uma vai te mostrar um ângulo novo sobre como um algoritmo pode "pensar" para separar classes.

O **Naive Bayes** pensa em termos de probabilidade: dado o que eu sei sobre esse ponto, qual é a chance de ele pertencer a cada classe? Já o **SVM** (Support Vector Machine, ou Máquina de Vetores de Suporte) pensa geometricamente: onde posso desenhar a fronteira que separa as classes com a maior margem possível?

São duas filosofias opostas, e as duas funcionam. Aprender quando usar cada uma é o que separa um cientista de dados iniciante de um experiente.

O dataset principal desta aula é o **Wine** (classificação de vinhos por características químicas), com demonstrações extras em dados artificiais para deixar os conceitos visuais e intuitivos.

---

## Como o projeto está organizado

```
aula_05/
│
├── app_streamlit.py                    # Aplicação web interativa — abre no navegador
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
streamlit run app_streamlit.py --server.port 8505
```

Acesse `http://localhost:8505` no navegador. Se estiver usando o portal principal, a navegação entre aulas já está configurada.

---

## O que você vai aprender — 3 seções

| # | Seção | O que é trabalhado |
|---|-------|--------------------|
| 1 | **Naive Bayes** | O Teorema de Bayes, distribuição Gaussiana, as variações do algoritmo (GaussianNB, MultinomialNB, BernoulliNB) e aplicação no dataset Wine com validação cruzada cv=10 |
| 2 | **SVM — Support Vector Machine** | Hiperplano e margem máxima, parâmetro C (penalidade), kernel RBF e parâmetro gamma, demonstração interativa da fronteira de decisão em 2D e GridSearchCV para otimização |
| 3 | **Comparativo Final** | KNN vs Naive Bayes vs SVM no dataset Wine — quem ganha? Por quê? |

---

## O que dá pra mexer em tempo real

- **Distribuição Gaussiana interativa** → ajuste a média e o desvio padrão e veja como a curva muda
- **Fronteira de decisão do SVM** → troque o kernel (linear, RBF, polinomial) e veja as fronteiras mudar de forma
- **Parâmetro C** → aumente e diminua e veja o tradeoff entre margem larga e erros permitidos
- **Parâmetro Gamma (γ)** → controle o "raio de influência" de cada ponto de suporte no kernel RBF
- **GridSearchCV** → explore a tabela completa da busca de hiperparâmetros e entenda qual combinação ganhou

---

## Os conceitos principais

### Naive Bayes — pensando em probabilidades

A ideia vem do **Teorema de Bayes**: a probabilidade de uma hipótese dado uma evidência. Aplicado à classificação, queremos saber: dado que um vinho tem essas características químicas, qual a probabilidade de ser da classe 1, 2 ou 3?

$$P(C_k \mid X) = \frac{P(X \mid C_k) \cdot P(C_k)}{P(X)}$$

O "Naive" (ingênuo) do nome vem da suposição simplificadora de que todas as features são **independentes entre si** dado a classe. Isso raramente é verdade na prática, mas o algoritmo ainda assim tende a funcionar bem.

O **GaussianNB** (Naive Bayes Gaussiano) assume que cada feature segue uma distribuição normal dentro de cada classe. Ele estima a média e o desvio padrão de cada feature para cada classe e usa isso para calcular as probabilidades.

Use o Naive Bayes quando tiver muitas features, trabalhar com texto, ou quando a velocidade de treino importar mais que a precisão máxima.

### SVM — pensando geometricamente

O SVM encontra o **hiperplano** que separa as classes com a **maior margem possível**. Os pontos mais próximos da fronteira são chamados de **vetores de suporte**, e são os únicos que importam para definir a separação.

O parâmetro **C** controla o equilíbrio entre margem larga e erros:
- C pequeno → margem larga, mais erros tolerados (modelo mais simples, menos propenso a overfitting)
- C grande → margem estreita, menos erros tolerados (modelo mais complexo, pode decorar o treino)

**Kernels** permitem que o SVM classifique dados que não são separáveis por uma linha reta. O kernel projeta os dados em uma dimensão maior onde a separação se torna possível:
- **Linear** → reta simples (rápido, bom para dados linearmente separáveis)
- **RBF** (Radial Basis Function) → curvas suaves (o mais usado na prática)
- **Polinomial** → fronteiras polinomiais (útil em casos específicos)

---

## GridSearchCV — encontrando os melhores hiperparâmetros

Em vez de testar C e gamma manualmente, o `GridSearchCV` testa todas as combinações automaticamente e entrega a melhor configuração:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "svc__C":     [0.1, 1, 10, 100],
    "svc__gamma": ["scale", 0.001, 0.01, 0.1],
    "svc__kernel": ["rbf", "linear"],
}

grid = GridSearchCV(pipeline, param_grid, cv=10,
                    scoring="f1_weighted", n_jobs=-1)
grid.fit(X_train, y_train)

print("Melhores parâmetros:", grid.best_params_)
```

O `n_jobs=-1` usa todos os núcleos do processador em paralelo — o que acelera muito a busca.

---

## As métricas de avaliação

| Métrica | O que mede | Como interpretar |
|---------|-----------|-----------------|
| **Accuracy** | % de previsões corretas | Boa para dados balanceados |
| **F1-Score** | Equilíbrio entre precision e recall | Ideal para dados desbalanceados |
| **Validação Cruzada (cv=10)** | Desempenho médio em 10 divisões diferentes | Mais confiável que um único split |
| **Matriz de Confusão** | Tabela de acertos e erros por classe | Mostra exatamente onde o modelo erra |

---

## As ferramentas do projeto

| Biblioteca | O que faz nesta aula |
|------------|---------------------|
| **scikit-learn** | `GaussianNB`, `SVC`, `Pipeline`, `StandardScaler`, `GridSearchCV`, `StratifiedKFold`, `cross_val_score`, `load_wine`, `make_blobs`, `make_moons` |
| **pandas** | Manipulação dos datasets e tabelas de resultados |
| **matplotlib** | Fronteiras de decisão 2D, curvas de distribuição Gaussiana |
| **seaborn** | Heatmap da matriz de confusão, visualizações do dataset Wine |

---

## O dataset utilizado

### Wine — Classificação de Vinhos

O **Wine dataset** contém 178 amostras de vinhos italianos de três produtores diferentes (classes 0, 1 e 2), com 13 características químicas medidas em cada amostra.

| Feature | Descrição |
|---------|-----------|
| `alcohol` | Teor alcoólico |
| `malic_acid` | Ácido málico |
| `ash` | Cinzas |
| `flavanoids` | Flavonoides |
| `color_intensity` | Intensidade da cor |
| `proline` | Prolina |
| ... | 13 features no total |

É um dataset clássico, balanceado e desafiador o suficiente para mostrar diferenças reais entre os algoritmos.

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
