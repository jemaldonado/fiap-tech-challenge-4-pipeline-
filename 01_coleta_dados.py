"""
ETAPA 1: Coleta e Preparação dos Dados
Objetivo: Baixar dados históricos de preços e preparar para treino
Suporta múltiplos tickers simultaneamente
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import pickle

# Configuração dos dados
SYMBOLS = ['NVDA', 'MELI', 'NU']
START_DATE = '2015-01-01'  # Começar em 2015 para mais histórico
END_DATE = '2026-05-19'
BASE_DATA_DIR = './data/raw'

os.makedirs(BASE_DATA_DIR, exist_ok=True)

print("=" * 60)
print("ETAPA 1: COLETA E PREPARACAO DE DADOS")
print("=" * 60)

for SYMBOL in SYMBOLS:
    print(f"\n{'='*60}")
    print(f"Processando: {SYMBOL}")
    print(f"{'='*60}")

    DATA_DIR = f'{BASE_DATA_DIR}/{SYMBOL}'
    os.makedirs(DATA_DIR, exist_ok=True)

    try:
        print(f"Baixando dados de {START_DATE} a {END_DATE}...")
        df = yf.download(SYMBOL, start=START_DATE, end=END_DATE, progress=False)

        if df.empty or len(df) < 100:
            print(f"Erro: Não há dados suficientes ({len(df)} registros)")
            continue

        print(f"Sucesso: {len(df)} pregões capturados")
    except Exception as e:
        print(f"Erro ao baixar: {e}")
        continue

    print(f"\nDados capturados:")
    print(f"  Total: {df.shape[0]} pregões")
    print(f"  Período: {df.index[0].date()} a {df.index[-1].date()}")
    print(f"  Colunas: {', '.join(df.columns)}")
    print(f"  Valores faltando: {df.isnull().sum().sum()}")

    if isinstance(df.columns, pd.MultiIndex):
        data = df[('Close', SYMBOL)].values.reshape(-1, 1)
        close_series = df[('Close', SYMBOL)]
    else:
        data = df[['Close']].values
        close_series = df['Close']

    print(f"\nEstatísticas de preço:")
    print(f"  Mínimo: ${close_series.min():.2f}")
    print(f"  Máximo: ${close_series.max():.2f}")
    print(f"  Média: ${close_series.mean():.2f}")
    print(f"  Última: ${close_series.iloc[-1]:.2f}")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    print(f"\nNormalizado para escala 0-1")
    print(f"  Amostra: {scaled_data[:3].flatten()}")

    np.save(f'{DATA_DIR}/scaled_prices.npy', scaled_data)

    with open(f'{DATA_DIR}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    df.to_csv(f'{DATA_DIR}/raw_prices.csv')

    metadata = {
        'symbol': SYMBOL,
        'n_records': len(df),
        'start_date': str(df.index[0].date()),
        'end_date': str(df.index[-1].date()),
        'min_price': float(close_series.min()),
        'max_price': float(close_series.max()),
        'last_price': float(close_series.iloc[-1])
    }

    with open(f'{DATA_DIR}/metadata.txt', 'w') as f:
        for k, v in metadata.items():
            f.write(f"{k}: {v}\n")

    print(f"\nSalvo em {DATA_DIR}/")
    print(f"  - scaled_prices.npy")
    print(f"  - scaler.pkl")
    print(f"  - raw_prices.csv")

print("\n" + "=" * 60)
print("ETAPA 1 CONCLUIDA COM SUCESSO")
print(f"Total de tickers processados: {len(SYMBOLS)}")
print("=" * 60)
