"""
ETAPA 5: Otimizacao de Hiperparametros com Grid Search
Testa combinacoes de hiperparametros e encontra os melhores
"""

import numpy as np
import json
import os
import sys
from datetime import datetime
from itertools import product
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

SYMBOLS = ['NVDA', 'MELI', 'NU']

WINDOW_SIZES = [60, 120, 180]
LEARNING_RATES = [0.0001, 0.0005, 0.001]
DROPOUTS = [0.1, 0.2, 0.3]
BATCH_SIZE = 32
EPOCHS = 500

print("=" * 80)
print("ETAPA 5: OTIMIZACAO DE HIPERPARAMETROS COM GRID SEARCH")
print("=" * 80)
print(f"\nParametros a testar:")
print(f"  WINDOW_SIZES:    {WINDOW_SIZES}")
print(f"  LEARNING_RATES:  {LEARNING_RATES}")
print(f"  DROPOUTS:        {DROPOUTS}")
print(f"  Total combinacoes: {len(WINDOW_SIZES) * len(LEARNING_RATES) * len(DROPOUTS)} × {len(SYMBOLS)} tickers")
print(f"  Total modelos a treinar: {len(WINDOW_SIZES) * len(LEARNING_RATES) * len(DROPOUTS) * len(SYMBOLS)}")
print("=" * 80 + "\n")

resultados_globais = {}

def criar_modelo(learning_rate, dropout):
    """Cria modelo LSTM com hiperparametros especificados"""
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=(None, 1)),
        Dropout(dropout),
        LSTM(32, activation='relu'),
        Dropout(dropout),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    return model

def treinar_e_avaliar(symbol, window_size, learning_rate, dropout):
    """Treina modelo e retorna metricas de avaliacao"""
    print(f"\n{'='*60}")
    print(f"Treinando: {symbol} | Window={window_size} | LR={learning_rate} | Dropout={dropout}")
    print(f"{'='*60}")

    DATA_PROCESSED_DIR = f'./data/processed/{symbol}'
    MODELS_DIR = f'./models/{symbol}'
    os.makedirs(MODELS_DIR, exist_ok=True)

    try:
        X_train = np.load(f'{DATA_PROCESSED_DIR}/X_train.npy')
        y_train = np.load(f'{DATA_PROCESSED_DIR}/y_train.npy')
        X_val = np.load(f'{DATA_PROCESSED_DIR}/X_val.npy')
        y_val = np.load(f'{DATA_PROCESSED_DIR}/y_val.npy')
        X_test = np.load(f'{DATA_PROCESSED_DIR}/X_test.npy')
        y_test = np.load(f'{DATA_PROCESSED_DIR}/y_test.npy')

        if X_train.shape[1] != window_size:
            print(f"  [SKIP] Window size {window_size} nao corresponde aos dados ({X_train.shape[1]})")
            return None

        model = criar_modelo(learning_rate, dropout)

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=0
        )

        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val),
            callbacks=[early_stop],
            verbose=0
        )

        y_pred = model.predict(X_test, verbose=0).flatten()

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        directional_accuracy = np.mean(
            np.sign(y_test[1:] - y_test[:-1]) ==
            np.sign(y_pred[1:] - y_pred[:-1])
        )

        epochs_treinado = len(history.history['loss'])

        resultado = {
            'symbol': symbol,
            'window_size': window_size,
            'learning_rate': learning_rate,
            'dropout': dropout,
            'epochs_treinado': epochs_treinado,
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'mape': float(mape),
            'directional_accuracy': float(directional_accuracy)
        }

        print(f"  [OK] MAPE={mape:.2f}% | R²={r2:.4f} | Dir.Acc={directional_accuracy*100:.1f}%")
        return resultado

    except Exception as e:
        print(f"  [ERRO] {str(e)}")
        return None

start_time = datetime.now()

for symbol in SYMBOLS:
    print(f"\n\n{'#'*80}")
    print(f"# {symbol}")
    print(f"{'#'*80}")

    resultados_ticker = []

    for window_size in WINDOW_SIZES:
        for learning_rate in LEARNING_RATES:
            for dropout in DROPOUTS:
                resultado = treinar_e_avaliar(symbol, window_size, learning_rate, dropout)
                if resultado:
                    resultados_ticker.append(resultado)

    resultados_globais[symbol] = resultados_ticker

    if resultados_ticker:
        melhor = max(resultados_ticker, key=lambda x: x['r2'])
        print(f"\nMelhor configuracao para {symbol}:")
        print(f"  Window={melhor['window_size']} | LR={melhor['learning_rate']} | Dropout={melhor['dropout']}")
        print(f"  MAPE={melhor['mape']:.2f}% | R²={melhor['r2']:.4f} | Dir.Acc={melhor['directional_accuracy']*100:.1f}%")

tempo_total = (datetime.now() - start_time).total_seconds() / 60

print(f"\n\n{'='*80}")
print("RESUMO FINAL")
print(f"{'='*80}")

melhor_por_ticker = {}
for symbol, resultados in resultados_globais.items():
    if resultados:
        melhor = max(resultados, key=lambda x: x['r2'])
        melhor_por_ticker[symbol] = melhor
        print(f"\n{symbol}:")
        print(f"  Melhor: Window={melhor['window_size']} | LR={melhor['learning_rate']} | Dropout={melhor['dropout']}")
        print(f"  MAPE={melhor['mape']:.2f}% | R²={melhor['r2']:.4f} | Dir.Acc={melhor['directional_accuracy']*100:.1f}%")

arquivo_saida = './APRESENTACAO/resultados_otimizacao.json'
os.makedirs('./APRESENTACAO', exist_ok=True)

with open(arquivo_saida, 'w', encoding='utf-8') as f:
    json.dump({
        'timestamp': datetime.now().isoformat(),
        'tempo_total_minutos': tempo_total,
        'melhor_por_ticker': melhor_por_ticker,
        'todos_resultados': resultados_globais
    }, f, indent=2, ensure_ascii=False)

print(f"\n[OK] Resultados salvos em: {arquivo_saida}")
print(f"[OK] Tempo total: {tempo_total:.1f} minutos")
print(f"{'='*80}\n")

print("Proxima etapa: retreinar modelos finais com melhores hiperparametros")
