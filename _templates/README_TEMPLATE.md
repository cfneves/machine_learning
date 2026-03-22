# {EMOJI} Aula {NN} — {Tema Completo}

<!--
  INSTRUÇÕES DE USO:
  Substitua todos os campos entre chaves {} pelo conteúdo da aula.
  Remova seções marcadas como [OPCIONAL] se não se aplicarem.
  Não altere a ordem das seções.
-->

<div align="center">

**Cláudio Ferreira Neves  ·  Especialista em Ciência de Dados e IA**

</div>

> Material didático autoral. Reprodução ou uso sem crédito ao autor é considerado plágio.

---

## Sobre esta aula

<!--
  3–5 linhas contextualizando o tema.
  Responda: por que esse assunto existe? Que problemas ele resolve?
  Cite o(s) dataset(s) utilizado(s) no final.
-->

{Contexto motivacional da aula — por que esse tema importa, onde aparece no mundo real.}

Os datasets desta aula são: **{Dataset 1}** {descrição breve} e **{Dataset 2}** {descrição breve}.

---

## Como o projeto está organizado

```
aula_{NN}/
│
├── app_streamlit.py                   # Aplicação web interativa — abre no navegador
├── aula{NN}_{tema}.py                 # Script Python principal — roda no terminal
├── Aula_{NN}_{Tema}_(resolvido).ipynb # Notebook completo com soluções
├── Aula_{NN}_{Tema}_(aluno).ipynb     # Notebook com exercícios para praticar
├── Slides_Aula_{NN}_{Tema}.pdf        # Material teórico da aula
├── requirements.txt                   # Bibliotecas necessárias
├── README.md                          # Este arquivo
│
└── outputs/                           # Gráficos salvos automaticamente (150 DPI)
    ├── 01_{descricao}.png
    ├── 02_{descricao}.png
    └── ...
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
pip install {lista de bibliotecas separadas por espaço}
```

### 2. Abra a aplicação no navegador

```bash
streamlit run app_streamlit.py --server.port {850N}
```

Acesse `http://localhost:{850N}` no navegador. Se estiver usando o portal principal, a navegação entre aulas já está configurada.

### 3. Rodar o script Python puro (sem interface)

```bash
python aula{NN}_{tema}.py
```

Os resultados aparecem no terminal e os gráficos são salvos na pasta `outputs/`.

---

## O que você vai aprender — {N} seções

| # | Seção | O que é trabalhado |
|---|-------|--------------------|
| 1 | **{Nome da Seção 1}** | {Descrição do que é feito e aprendido} |
| 2 | **{Nome da Seção 2}** | {Descrição do que é feito e aprendido} |
| 3 | **{Nome da Seção 3}** | {Descrição do que é feito e aprendido} |
| 4 | **{Nome da Seção 4}** | {Descrição do que é feito e aprendido} |

---

## O que dá pra mexer em tempo real

<!-- [OPCIONAL] Remova esta seção se a aula não tiver app Streamlit com controles interativos -->

- **{Controle 1}** → {O que acontece ao ajustar}
- **{Controle 2}** → {O que acontece ao ajustar}
- **{Controle 3}** → {O que acontece ao ajustar}

---

## Os conceitos principais

<!--
  Um bloco por algoritmo/conceito central da aula.
  Estrutura: nome em negrito → explicação em linguagem simples → fórmula LaTeX (se houver).
  Máximo 4–5 linhas por conceito. Use analogias do cotidiano.
-->

### {Conceito 1}

{Explicação simples em 2–3 linhas. Use analogia se o conceito for abstrato.}

$$\hat{y} = {formula LaTeX}$$

Onde **{variável}** é {o que representa} e **{variável}** é {o que representa}.

### {Conceito 2}

{Explicação simples em 2–3 linhas.}

$$P(y=1 \mid X) = {formula LaTeX}$$

---

## As métricas de avaliação

<!-- [OPCIONAL] Use para aulas de regressão ou classificação -->

| Métrica | O que mede | Como interpretar |
|---------|-----------|-----------------|
| **{Métrica 1}** | {O que ela mede} | {Como ler o resultado — ex: "quanto mais perto de 1, melhor"} |
| **{Métrica 2}** | {O que ela mede} | {Como ler o resultado} |
| **{Métrica 3}** | {O que ela mede} | {Como ler o resultado} |

---

## Os datasets utilizados

<!-- [OPCIONAL] Use quando houver mais de um dataset ou quando o dataset precisar de apresentação -->

| Dataset | Registros | Variável alvo | Objetivo nesta aula |
|---------|-----------|---------------|---------------------|
| **{Dataset 1}** | {N} | {coluna alvo} | {Por que foi escolhido / o que demonstra} |
| **{Dataset 2}** | {N} | {coluna alvo} | {Por que foi escolhido / o que demonstra} |

---

## As ferramentas do projeto

| Biblioteca | O que faz nesta aula |
|------------|---------------------|
| **{lib 1}** | {Uso específico nesta aula — não genérico} |
| **{lib 2}** | {Uso específico nesta aula} |
| **{lib 3}** | {Uso específico nesta aula} |
| **{lib 4}** | {Uso específico nesta aula} |

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
