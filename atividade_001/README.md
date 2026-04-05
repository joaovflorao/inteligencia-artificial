# README — Atividade 001: Penguins

**Disciplina:** Inteligência Artificial  
**Professor:** Marcelo Batista

---

## Dataset escolhido

`penguins` (Seaborn) — medições morfológicas de pinguins coletadas na Antártida (Gorman et al., 2014).

- **344 linhas × 7 colunas**
- **Target:** `species` (classificação multiclasse: `Adelie`, `Chinstrap`, `Gentoo`)

---

## Definição do problema

| Item | Detalhe |
|---|---|
| **Target (y)** | `species` |
| **Tipo** | Classificação multiclasse (3 classes) |
| **Features numéricas** | `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g` |
| **Features categóricas** | `island`, `sex` |
| **Colunas removidas** | Nenhuma |

---

## Diagnóstico dos dados

- **Missing values:** `sex` (17 faltantes, ~5%), numéricas (3 faltantes cada, ~0.9%)
- **Distribuição do target:** Adelie 44% · Gentoo 36% · Chinstrap 20% (leve desbalanceamento)
- **Necessidades:** imputação (missing) + codificação (categóricas)

---

## Pipeline

- **Numéricas:** `SimpleImputer(median)` → `StandardScaler`
- **Categóricas:** `SimpleImputer(most_frequent)` → `OneHotEncoder`
- **Split:** 80% treino / 20% teste, estratificado, `random_state=42`
- O pré-processamento **aprende apenas no treino** e é aplicado no teste

---

## Modelos e resultados

| Modelo | Accuracy | Erros no teste |
|---|---|---|
| **Baseline** — LogisticRegression | **1.0000** | 0 |
| **Melhoria** — RandomForestClassifier | 0.9710 | 2 |

### Matriz de confusão — LogisticRegression

|  | Adelie | Chinstrap | Gentoo |
|---|---|---|---|
| **Adelie** | 30 | 0 | 0 |
| **Chinstrap** | 0 | 14 | 0 |
| **Gentoo** | 0 | 0 | 25 |

### Matriz de confusão — RandomForestClassifier

|  | Adelie | Chinstrap | Gentoo |
|---|---|---|---|
| **Adelie** | 29 | 1 | 0 |
| **Chinstrap** | 1 | 13 | 0 |
| **Gentoo** | 0 | 0 | 25 |

---

## Interpretação dos erros

- **Baseline:** perfeito no teste — as três espécies são linearmente separáveis nas features utilizadas.
- **Melhoria (RF):** 2 erros, ambos na fronteira **Adelie ↔ Chinstrap** — as duas espécies têm `bill_depth_mm` muito próximos (~18 mm), dificultando a separação para indivíduos atípicos.
- **Gentoo** nunca é confundido: sua morfologia (maior massa corporal, nadadeira mais longa) é muito distinta das demais.
- O baseline superou a melhoria neste caso — resultado legítimo: com classes bem separadas e dataset pequeno, modelos simples podem generalizar melhor.

---

## Comparação técnica dos modelos

- **Baseline (LogisticRegression):** modelo simples, interpretável, assume fronteira linear. Bom ponto de partida.
- **Melhoria (RandomForestClassifier):** ensemble de árvores, captura não-linearidades. Leve overfitting no treino com dataset pequeno explicaria o desempenho menor no teste.
- **Missing values:** numéricas com mediana; categóricas com moda (`SimpleImputer` no Pipeline).
- **Comparação justa:** somente o classificador muda — pré-processamento idêntico nos dois modelos.

---

## Limitação e melhoria sugerida (ponto extra)

**Limitação:** dataset pequeno (344 amostras) com desbalanceamento moderado em `Chinstrap` (~20%). Uma única divisão treino/teste pode não ser representativa da performance real.

**Melhoria sugerida:** usar **validação cruzada estratificada** (`StratifiedKFold`, k=5 ou k=10) para obter uma estimativa mais robusta e com menor variância.
