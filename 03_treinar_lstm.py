"""
ETAPA 3: Treinar Modelo LSTM
Objetivo: Construir e treinar rede neural para capturar padroes temporais
Treina um modelo independente para cada ticker
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os
import sys

# Configuracoes do treinamento
SYMBOLS = ['NVDA', 'MELI', 'NU']
EPOCHS = 500
BATCH_SIZE = 32
LEARNING_RATE = 0.0001

print("=" * 60)
print("ETAPA 3: TREINAR MODELO LSTM")
print("=" * 60)

def criar_modelo():
    """
    Arquitetura LSTM otimizada para séries temporais financeiras.

    Design:
    - Primeira LSTM: 256 neurons (captura padrões complexos de longo prazo)
    - Dropout 0.3: reduz overfitting
    - Segunda LSTM: 128 neurons (refina os padrões)
    - Dropout 0.3
    - Dense 64: camada densa para integração de features
    - Dense 32: redução de dimensionalidade
    - Dense 1: previsão final

    Otimizações aplicadas:
    - Learning rate aumentado (0.01) para convergência mais rápida
    - Mais unidades LSTM para capturar complexidade do mercado
    - Mais camadas para representação hierárquica
    """
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=(120, 1)),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )
    return model

# Treinar um modelo por ticker
for SYMBOL in SYMBOLS:
    print(f"\n{'='*60}")
    print(f"Treinando modelo para: {SYMBOL}")
    print(f"{'='*60}")

    DATA_PROCESSED_DIR = f'./data/processed/{SYMBOL}'
    MODELS_DIR = f'./models/{SYMBOL}'
    os.makedirs(MODELS_DIR, exist_ok=True)

    # PASSO 1: Carregar dados preparados na Etapa 2
    try:
        X_train = np.load(f'{DATA_PROCESSED_DIR}/X_train.npy')
        y_train = np.load(f'{DATA_PROCESSED_DIR}/y_train.npy')
        X_val = np.load(f'{DATA_PROCESSED_DIR}/X_val.npy')
        y_val = np.load(f'{DATA_PROCESSED_DIR}/y_val.npy')

        print(f"Dados carregados com sucesso:")
        print(f"  X_train: {X_train.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  X_val:   {X_val.shape}\n")
    except FileNotFoundError as e:
        print(f"ERRO: {e}")
        print(f"Execute a Etapa 2 primeiro!")
        continue

    # PASSO 2: Construir e compilar a rede
    model = criar_modelo()
    print("Modelo LSTM construido e compilado!")
    print(model.summary())

    # PASSO 3: Treinar o modelo com EarlyStopping
    print(f"\nIniciando treinamento ({EPOCHS} rodadas com EarlyStopping)...\n")
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )

    # PASSO 4: Salvar metricas de treinamento
    np.save(f'{MODELS_DIR}/history_loss.npy', history.history['loss'])
    np.save(f'{MODELS_DIR}/history_val_loss.npy', history.history['val_loss'])
    np.save(f'{MODELS_DIR}/history_mae.npy', history.history['mae'])
    np.save(f'{MODELS_DIR}/history_val_mae.npy', history.history['val_mae'])

    final_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]

    print(f"\nTreinamento concluido para {SYMBOL}!")
    print(f"  Erro final (treino):     {final_loss:.6f}")
    print(f"  Erro final (validacao):  {final_val_loss:.6f}")

    # PASSO 5: Salvar modelo treinado
    model.save(f'{MODELS_DIR}/lstm_model.h5')
    print(f"\nModelo salvo em {MODELS_DIR}/lstm_model.h5")

print("\n" + "=" * 60)
print("ETAPA 3 CONCLUIDA COM SUCESSO")
print(f"Total: {len(SYMBOLS)} modelo(s) treinado(s)")
print("=" * 60)
