"""
ETAPA 6: Retreinar Modelos com Melhores Hiperparametros
Apos otimizacao, retreina os modelos finais com melhores parametros
"""

import numpy as np
import json
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

SYMBOLS = ['NVDA', 'MELI', 'NU']
BATCH_SIZE = 32
EPOCHS = 500

print("=" * 80)
print("ETAPA 6: RETREINAR COM MELHORES HIPERPARAMETROS")
print("=" * 80)

resultado_otimizacao_path = './APRESENTACAO/resultados_otimizacao.json'

if not os.path.exists(resultado_otimizacao_path):
    print(f"\nERRO: {resultado_otimizacao_path} nao encontrado!")
    print("Execute a otimizacao (05_otimizar_hiperparametros.py) primeiro.")
    exit(1)

with open(resultado_otimizacao_path, 'r', encoding='utf-8') as f:
    dados_otimizacao = json.load(f)

melhores_hp = dados_otimizacao['melhor_por_ticker']

print(f"\nMelhores Hiperparametros encontrados:")
for symbol, config in melhores_hp.items():
    print(f"\n{symbol}:")
    print(f"  Window Size: {config['window_size']}")
    print(f"  Learning Rate: {config['learning_rate']}")
    print(f"  Dropout: {config['dropout']}")
    print(f"  Metricas: MAPE={config['mape']:.2f}% | R²={config['r2']:.4f}")

print(f"\n{'='*80}")
print("Iniciando retreinamento final...\n")

def criar_modelo(learning_rate, dropout):
    """Cria modelo LSTM"""
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

for symbol in SYMBOLS:
    if symbol not in melhores_hp:
        print(f"\n[SKIP] {symbol}: nao foi otimizado")
        continue

    config = melhores_hp[symbol]
    window_size = config['window_size']
    learning_rate = config['learning_rate']
    dropout = config['dropout']

    print(f"\n{'='*60}")
    print(f"Retreinando {symbol}")
    print(f"  Window={window_size} | LR={learning_rate} | Dropout={dropout}")
    print(f"{'='*60}")

    DATA_PROCESSED_DIR = f'./data/processed/{symbol}'
    MODELS_DIR = f'./models/{symbol}'
    os.makedirs(MODELS_DIR, exist_ok=True)

    try:
        X_train = np.load(f'{DATA_PROCESSED_DIR}/X_train.npy')
        y_train = np.load(f'{DATA_PROCESSED_DIR}/y_train.npy')
        X_val = np.load(f'{DATA_PROCESSED_DIR}/X_val.npy')
        y_val = np.load(f'{DATA_PROCESSED_DIR}/y_val.npy')

        if X_train.shape[1] != window_size:
            print(f"  [SKIP] Window size mismatch: esperado {window_size}, encontrado {X_train.shape[1]}")
            continue

        model = criar_modelo(learning_rate, dropout)

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )

        print(f"\nTreinando ({EPOCHS} epochs com EarlyStopping)...")
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val),
            callbacks=[early_stop],
            verbose=1
        )

        final_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]

        print(f"\n[OK] Treinamento concluido!")
        print(f"  Loss final (treino):    {final_loss:.6f}")
        print(f"  Loss final (validacao): {final_val_loss:.6f}")

        model.save(f'{MODELS_DIR}/lstm_model.h5')
        print(f"  Modelo salvo em: {MODELS_DIR}/lstm_model.h5")

        np.save(f'{MODELS_DIR}/history_loss.npy', history.history['loss'])
        np.save(f'{MODELS_DIR}/history_val_loss.npy', history.history['val_loss'])
        np.save(f'{MODELS_DIR}/history_mae.npy', history.history['mae'])
        np.save(f'{MODELS_DIR}/history_val_mae.npy', history.history['val_mae'])

        config_final = {
            'symbol': symbol,
            'window_size': window_size,
            'learning_rate': learning_rate,
            'dropout': dropout,
            'batch_size': BATCH_SIZE,
            'epochs_treinado': len(history.history['loss']),
            'final_loss': float(final_loss),
            'final_val_loss': float(final_val_loss),
            'nota': 'Modelo retreinado com melhores hiperparametros da otimizacao'
        }

        with open(f'{MODELS_DIR}/config_final.json', 'w', encoding='utf-8') as f:
            json.dump(config_final, f, indent=2, ensure_ascii=False)

        print(f"  Configuracao salva em: {MODELS_DIR}/config_final.json")

    except Exception as e:
        print(f"  [ERRO] {str(e)}")
        continue

print(f"\n{'='*80}")
print("RETREINAMENTO CONCLUIDO!")
print(f"{'='*80}")
print("\nModelos otimizados prontos para deploy")
print("Proxima etapa: copiar para C:\\Users\\juan\\estudo-agents\\lstm-predictor")
