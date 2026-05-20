# FIAP Tech Challenge 4 - Pipeline ML

Pipeline completo para coleta, processamento, treinamento e avaliação de modelos LSTM para previsão de preços de ações.

## Visão Geral

Este projeto implementa um pipeline de aprendizado de máquina que automatiza o processo de previsão de preços de ações usando redes neurais LSTM (Long Short-Term Memory). O pipeline coleta dados históricos do Yahoo Finance, normaliza os dados, treina modelos otimizados e gera métricas de avaliação.

Os modelos treinados aqui são utilizados pela API em produção: https://github.com/jemaldonado/fiap-tech-challenge-4

## Arquitetura do Pipeline

O pipeline é dividido em 4 etapas principais, cada uma em um arquivo Python separado:

### Etapa 1: Coleta de Dados (01_coleta_dados.py)
Faz o download de dados históricos de preços de ações do Yahoo Finance para os tickers NVDA, MELI e NU. Os dados são armazenados em formato CSV para cada ticker, cobrindo um período de aproximadamente 11 anos. Gera também os scalers (MinMaxScaler) que serão usados na normalização.

Saída: `data/raw/{SYMBOL}/raw_prices.csv` e `data/raw/{SYMBOL}/scaler.pkl`

### Etapa 2: Preparação de Sequências (02_preparar_sequencias.py)
Transforma os dados brutos em sequências de tempo adequadas para o treinamento de redes LSTM. Define uma janela de 120 dias de histórico para prever o preço do dia seguinte. Normaliza os dados usando MinMaxScaler e os divide em conjuntos de treino (80%) e teste (20%).

Saída: `data/processed/{SYMBOL}/train.pkl` e `data/processed/{SYMBOL}/test.pkl`

### Etapa 3: Treinamento LSTM (03_treinar_lstm.py)
Treina modelos LSTM com a arquitetura padrão: duas camadas LSTM de 50 unidades cada, seguidas por uma camada densa. Usa Grid Search para otimizar hiperparâmetros (testa 27 combinações diferentes de batch size, épocas e taxa de aprendizado). Salva o melhor modelo para cada ticker.

Saída: `models/{SYMBOL}/lstm_model.h5`

### Etapa 4: Avaliação de Modelos (04_avaliar_modelo.py)
Avalia o desempenho dos modelos treinados no conjunto de teste. Calcula métricas como R², MAPE, MAE e RMSE nos valores reais (em USD, não normalizados). Computa também a acurácia direcional (percentual de vezes que o modelo acerta a direção do movimento). Salva um relatório com as métricas em formato JSON.

Saída: `models/{SYMBOL}/metrics.json`

## Requisitos e Instalação

### Dependências do Sistema
- Python 3.9 ou superior
- pip (gerenciador de pacotes Python)

### Instalação de Dependências Python

```bash
pip install -r requirements.txt
```

Principais bibliotecas:
- yfinance: coleta de dados do Yahoo Finance
- pandas: manipulação de dados
- numpy: computações numéricas
- scikit-learn: normalização e métricas
- tensorflow: framework de deep learning
- keras: API para construção de modelos

## Como Executar

### Executar o Pipeline Completo

```bash
python run_pipeline.py
```

Isso executa as 4 etapas em sequência: coleta, preparação, treinamento e avaliação.

### Executar Etapas Individuais

Você pode executar cada etapa separadamente:

```bash
python 01_coleta_dados.py
python 02_preparar_sequencias.py
python 03_treinar_lstm.py
python 04_avaliar_modelo.py
```

Recomenda-se executar nesta ordem, pois cada etapa depende da anterior.

### Tempo de Execução

- Coleta de dados: 2-3 minutos
- Preparação de sequências: < 1 minuto
- Treinamento (com Grid Search): 15-20 minutos (depende do hardware)
- Avaliação: < 1 minuto

Tempo total: aproximadamente 20-25 minutos em uma máquina com GPU moderada.

## Estrutura de Diretórios

```
.
├── 01_coleta_dados.py              # Etapa 1: coleta
├── 02_preparar_sequencias.py       # Etapa 2: preparação
├── 03_treinar_lstm.py              # Etapa 3: treinamento
├── 04_avaliar_modelo.py            # Etapa 4: avaliação
├── data/
│   ├── raw/
│   │   ├── NVDA/
│   │   │   ├── raw_prices.csv      # Dados brutos
│   │   │   └── scaler.pkl          # Normalizador
│   │   ├── MELI/
│   │   └── NU/
│   └── processed/
│       ├── NVDA/
│       │   ├── train.pkl           # Sequências treino
│       │   └── test.pkl            # Sequências teste
│       ├── MELI/
│       └── NU/
├── models/
│   ├── NVDA/
│   │   ├── lstm_model.h5           # Modelo treinado
│   │   └── metrics.json            # Métricas de avaliação
│   ├── MELI/
│   └── NU/
└── README.md
```

## Dados e Modelos

### Fonte de Dados
Os dados históricos são obtidos do Yahoo Finance via a biblioteca yfinance. Cobrem um período de 11 anos (janeiro de 2015 até maio de 2026) para os tickers NVDA, MELI e NU.

### Características dos Dados
- NVDA (NVIDIA): aproximadamente 2.750 dias de pregão
- MELI (Mercado Libre): aproximadamente 2.750 dias de pregão
- NU (Nu Holdings): aproximadamente 1.600 dias de pregão (stock mais nova)

### Modelos Treinados
Cada modelo é uma rede neural LSTM com a seguinte arquitetura:
- Entrada: sequência de 120 dias de preços normalizados
- Camada LSTM 1: 50 unidades
- Camada LSTM 2: 50 unidades
- Camada Dense: 1 unidade (previsão do próximo dia)
- Otimizador: Adam
- Função de perda: Mean Squared Error (MSE)

### Grid Search
O treinamento utiliza Grid Search com os seguintes hiperparâmetros:
- Batch size: [16, 32, 64]
- Épocas: [50, 100, 150]
- Taxa de aprendizado: [0.001, 0.0005, 0.0001]

Total: 3 × 3 × 3 = 27 combinações testadas para encontrar a melhor.

## Métricas de Desempenho

### Métricas Calculadas

**R² (Coeficiente de Determinação)**
Indica qual percentual da variação de preços o modelo consegue explicar. Varia de 0 a 1, onde 1 é perfeito. Os modelos atingem aproximadamente 0.83-0.87.

**MAPE (Mean Absolute Percentage Error)**
Erro percentual médio em relação ao preço real. NVDA: 44.57%, MELI: 9.94%, NU: 13.48%.

**MAE (Mean Absolute Error)**
Erro médio em dólares. NVDA: $32.82, MELI: $114.07, NU: $1.12.

**RMSE (Root Mean Squared Error)**
Raiz do erro quadrático médio. Penaliza mais os erros grandes.

**Directional Accuracy**
Percentual de vezes que o modelo acerta corretamente a direção do movimento (sobe ou desce). Varia entre 46% e 53% para nossos modelos.

## Interpretação dos Resultados

Os modelos apresentam bom desempenho em capturar a variação de preços (R² alto), mas com margem de erro significativa nas previsões absolutas (MAPE 10-45%). A acurácia direcional moderada (ligeiramente melhor que acaso) sugere que os modelos não devem ser usados como única fonte de decisão em trading.

O modelo para MELI apresenta o melhor desempenho geral (MAPE mais baixo), seguido por NVDA. O modelo para NU tem desempenho mais fraco, indicando que o LSTM não é adequado para stocks altamente voláteis e com pouco histórico.

## Integração com Produção

Os modelos treinados neste pipeline são utilizados pela API em produção: https://github.com/jemaldonado/fiap-tech-challenge-4

A API carrega os arquivos:
- `models/{SYMBOL}/lstm_model.h5`: modelo treinado
- `models/{SYMBOL}/metrics.json`: métricas
- `data/raw/{SYMBOL}/scaler.pkl`: normalizador

## Troubleshooting

### Erro ao baixar dados do Yahoo Finance
Se você receber um erro de conexão ao tentar baixar dados:
```
yfinance.exceptions.DownloadError
```

Possíveis soluções:
- Verificar conexão com a internet
- Tentar novamente (Yahoo Finance ocasionalmente tem limites de requisição)
- Aumentar o timeout em yfinance: `yf.download(..., timeout=30)`

### Erro de memória durante treinamento
Se receber `MemoryError` durante o treinamento:
- Reduzir o batch size (alterar em 03_treinar_lstm.py)
- Fechar outros programas para liberar RAM
- Usar GPU se disponível (configurável em TensorFlow)

### Modelos ou dados não encontrados
Certificar-se de que:
- Os arquivos foram executados em ordem (01, 02, 03, 04)
- Cada etapa completou sem erros
- Os diretórios data/ e models/ existem

