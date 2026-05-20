"""
ETAPA 2: Preparar Sequências Temporais
Objetivo: Converter preços em pares (entrada, saída) para LSTM
Exemplo: [dia1, dia2...dia60] → [dia61]
Suporta múltiplos tickers
"""

import numpy as np
import os
import sys

# Configuracoes de preparacao dos dados
WINDOW_SIZE = 120  # Usar 120 dias (~6 meses) para capturar trends maiores
SYMBOLS = ['NVDA', 'MELI', 'NU']

print("=" * 60)
print("ETAPA 2: PREPARAR SEQUENCIAS TEMPORAIS")
print("=" * 60)

def criar_sequencias(dados, tamanho_janela):
    """
    Transforma os preços históricos em pares entrada-saída.

    Exemplo com 120 dias:
      Entrada:  [preço_dia1, preço_dia2, ..., preço_dia120]
      Saída:    preço_dia121

    Isso permite que o LSTM aprenda padrões temporais.
    """
    X, y = [], []
    for i in range(len(dados) - tamanho_janela):
        X.append(dados[i:i+tamanho_janela])
        y.append(dados[i+tamanho_janela])
    return np.array(X), np.array(y)

# Processar cada ticker
for SYMBOL in SYMBOLS:
    print(f"\n{'='*60}")
    print(f"Processando: {SYMBOL}")
    print(f"{'='*60}")

    DATA_RAW_DIR = f'./data/raw/{SYMBOL}'
    DATA_PROCESSED_DIR = f'./data/processed/{SYMBOL}'

    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

    # ===== PASSO 1: Carregar dados normalizados =====
    try:
        scaled_data = np.load(f'{DATA_RAW_DIR}/scaled_prices.npy')
        print(f"Dados carregados com sucesso: {scaled_data.shape[0]} registros")
    except FileNotFoundError:
        print(f"ERRO: Arquivo nao encontrado: {DATA_RAW_DIR}/scaled_prices.npy")
        print(f"Execute a Etapa 1 primeiro!")
        continue

    # ===== PASSO 2: Criar sequências =====
    X, y = criar_sequencias(scaled_data, WINDOW_SIZE)

    if len(X) == 0:
        print(f"ERRO: Dados insuficientes para {SYMBOL} ({len(scaled_data)} < {WINDOW_SIZE})")
        continue

    print(f"Sequencias criadas com sucesso!")
    print(f"  X (entrada): {X.shape}")
    print(f"  y (saida):   {y.shape}")

    # ===== PASSO 3: Dividir em treino (70%), validacao (15%), teste (15%) =====
    n_samples = len(X)
    train_size = int(n_samples * 0.7)
    val_size = int(n_samples * 0.15)

    X_train = X[:train_size]
    y_train = y[:train_size]

    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]

    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]

    print(f"\nDivisao dos dados:")
    print(f"  Treino:     {X_train.shape[0]} amostras (70%)")
    print(f"  Validacao:  {X_val.shape[0]} amostras (15%)")
    print(f"  Teste:      {X_test.shape[0]} amostras (15%)")

    # ===== PASSO 4: Salvar =====
    np.save(f'{DATA_PROCESSED_DIR}/X_train.npy', X_train)
    np.save(f'{DATA_PROCESSED_DIR}/y_train.npy', y_train)
    np.save(f'{DATA_PROCESSED_DIR}/X_val.npy', X_val)
    np.save(f'{DATA_PROCESSED_DIR}/y_val.npy', y_val)
    np.save(f'{DATA_PROCESSED_DIR}/X_test.npy', X_test)
    np.save(f'{DATA_PROCESSED_DIR}/y_test.npy', y_test)

    print(f"\nArquivos salvos em {DATA_PROCESSED_DIR}/")

print("\n" + "=" * 60)
print("ETAPA 2 CONCLUIDA COM SUCESSO")
print("=" * 60)
