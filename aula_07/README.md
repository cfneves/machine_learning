# 🔵 Aula 07 — K-Means e PCA

<div align="center">

**Cláudio Ferreira Neves  ·  Especialista em Ciência de Dados e IA**

</div>

> Material didático autoral. Reprodução ou uso sem crédito ao autor é considerado plágio.

---

## O que é essa aula?

Todas as aulas anteriores trabalharam com **aprendizado supervisionado**: você tinha os dados e sabia a resposta certa. O modelo aprendia a partir do gabarito.

Mas e quando não há gabarito? Quando os dados chegam sem rótulos e você quer descobrir se existe algum padrão ou agrupamento natural? Isso é o **aprendizado não supervisionado**.

Esta aula cobre dois algoritmos dessa categoria: o **K-Means**, que agrupa dados em clusters sem precisar de rótulos, e o **PCA** (Principal Component Analysis), que reduz a dimensionalidade preservando o que mais importa nos dados.

Os datasets desta aula são o **Mall Customers** (segmentação de clientes de shopping) e o **Breast Cancer**, usado desta vez de forma não supervisionada.

---

## Como o projeto está organizado

```
aula_07/
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
streamlit run app_streamlit.py --server.port 8507
```

Acesse `http://localhost:8507` no navegador. Se estiver usando o portal principal, a navegação entre aulas já está configurada.

---

## O que você vai aprender — 3 seções

| # | Seção | O que é trabalhado |
|---|-------|--------------------|
| 1 | **K-Means** | Algoritmo passo a passo, demonstração interativa 2D, segmentação de clientes com Elbow Method e métricas de avaliação (Silhouette, Calinski-Harabasz, Davies-Bouldin) |
| 2 | **PCA** | Como e por que reduzir dimensionalidade, variância explicada por componente, visualização 2D do Breast Cancer com 30 features comprimidas |
| 3 | **PCA + K-Means combinados** | Reduzir o espaço com PCA primeiro e depois clusterizar — a combinação mais usada em projetos reais |

---

## O que dá pra mexer em tempo real

- **Número de clusters K** → veja os grupos se formando em tempo real no gráfico 2D
- **Número de iterações** → acompanhe o K-Means convergindo passo a passo
- **Elbow Method interativo** → encontre o K ideal através do gráfico de inércia
- **Número de componentes do PCA** → controle quanto da variância original você quer preservar
- **PCA + K-Means** → experimente diferentes valores de K no espaço reduzido e veja as métricas mudando

---

## Os conceitos principais

### K-Means — agrupando sem rótulos

O algoritmo funciona assim:

1. Escolha K pontos aleatórios como centróides iniciais
2. Atribua cada ponto ao centróide mais próximo
3. Recalcule a posição de cada centróide como a média dos pontos do seu grupo
4. Repita os passos 2 e 3 até os centróides pararem de se mover (convergência)

A **inércia** é a função de custo do K-Means. Ela mede a soma das distâncias quadráticas de cada ponto ao seu centróide mais próximo:

$$\text{Inércia} = \sum_{i=1}^{n} \min_{k} \| x_i - \mu_k \|^2$$

Quanto menor a inércia, mais compactos são os clusters. O problema é que ela diminui sempre quando K aumenta, então ela sozinha não é suficiente para escolher o K ideal.

**Como escolher o K ideal?**

O **Elbow Method** (Método do Cotovelo) plota a inércia para diferentes valores de K. O K ideal fica onde a curva "dobra como um cotovelo": a partir daí, adicionar mais clusters não reduz a inércia de forma significativa.

### Métricas de avaliação de clusters

Como não temos rótulos, avaliamos a qualidade dos clusters de outras formas:

| Métrica | O que mede | Como interpretar |
|---------|-----------|-----------------|
| **Silhouette Score** | Quão bem cada ponto se encaixa no seu cluster vs o cluster vizinho | De -1 a 1 — quanto mais perto de 1, melhor |
| **Calinski-Harabasz** | Razão entre dispersão entre clusters e dentro dos clusters | Quanto maior, melhor |
| **Davies-Bouldin** | Média da similaridade entre cada cluster e o mais parecido com ele | Quanto menor, melhor |

### PCA — vendo em menos dimensões sem perder a essência

Com 30 colunas, visualizar os dados diretamente é impossível. O **PCA** cria novas variáveis chamadas **componentes principais**, que são combinações lineares das originais, ordenadas por quanto da variância total cada uma captura.

A primeira componente captura a maior variância possível. A segunda captura a maior variância do que sobrou. E assim por diante.

**Variância explicada acumulada** é a métrica principal: se as 2 primeiras componentes explicam 80% da variância, você pode visualizar seus dados em 2D sem perder muita informação.

$$\text{Variância explicada da componente } k = \frac{\lambda_k}{\sum_i \lambda_i}$$

Onde $\lambda_k$ são os autovalores da matriz de covariância. A matemática pode ficar para depois. O que importa: o PCA encontra as direções no espaço que carregam mais informação dos dados.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_reduzido = pca.fit_transform(X_escalonado)

print(pca.explained_variance_ratio_)
# [0.44, 0.19] → as 2 componentes explicam 63% da variância total
```

---

## O dataset utilizado

### Mall Customers — Segmentação de Clientes

O **Mall Customers dataset** contém informações de 200 clientes de um shopping:

| Coluna | Descrição |
|--------|-----------|
| `CustomerID` | Identificador do cliente |
| `Gender` | Gênero |
| `Age` | Idade |
| `Annual Income (k$)` | Renda anual em milhares de dólares |
| `Spending Score (1-100)` | Índice de gastos atribuído pelo shopping |

O objetivo é agrupar clientes com comportamento parecido, sem nenhum rótulo. É um caso de uso direto: empresas fazem isso para segmentar campanhas de marketing.

---

## Por que PCA antes de K-Means?

Em datasets com muitas features, o K-Means sofre com a **maldição da dimensionalidade**: as distâncias entre pontos ficam muito parecidas em espaços de alta dimensão, e o algoritmo perde eficiência. Reduzir a dimensionalidade com PCA antes de clusterizar:

1. Acelera o treinamento
2. Remove ruído (variâncias pequenas que podem atrapalhar)
3. Permite visualizar os clusters em 2D

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
