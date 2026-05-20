"""
ETAPA 4: Avaliar Modelo
Objetivo: Calcular metricas profissionais (MAPE, R², Directional Accuracy)
Metricas reais usadas em projetos de FinTech e ML
"""

import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import pickle
import sys

# CONFIGURACAO
SYMBOLS = ['NVDA', 'MELI', 'NU']

# Se passar simbolo na linha de comando, processar so esse
if len(sys.argv) > 1:
    SYMBOLS = [sys.argv[1].upper()]

print("=" * 60)
print("ETAPA 4: AVALIAR MODELO")
print("=" * 60)

# Avaliar cada modelo
all_metrics = {}

for SYMBOL in SYMBOLS:
    print(f"\n{'='*60}")
    print(f"Avaliando: {SYMBOL}")
    print(f"{'='*60}")

    DATA_PROCESSED_DIR = f'./data/processed/{SYMBOL}'
    DATA_RAW_DIR = f'./data/raw/{SYMBOL}'
    MODELS_DIR = f'./models/{SYMBOL}'

    # PASSO 1: Carregar modelo treinado e dados de teste
    try:
        model = load_model(f'{MODELS_DIR}/lstm_model.h5')
        X_test = np.load(f'{DATA_PROCESSED_DIR}/X_test.npy')
        y_test = np.load(f'{DATA_PROCESSED_DIR}/y_test.npy')

        with open(f'{DATA_RAW_DIR}/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        print(f"Modelo e dados carregados com sucesso")
    except Exception as e:
        print(f"ERRO: {e}")
        continue

    # PASSO 2: Fazer predicoes no conjunto de teste
    y_pred = model.predict(X_test, verbose=0)

    print(f"Predicoes geradas: {y_pred.shape}")
    print(f"  Exemplo real:  {y_test[0, 0]:.6f}")
    print(f"  Exemplo pred:  {y_pred[0, 0]:.6f}")

    # PASSO 3: Calcular metricas em valores normalizados (0-1)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) * 100

    print(f"\nMetricas (valores normalizados 0-1):")
    print(f"  MAE:  {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAPE: {mape:.2f}%")

    # PASSO 4: Converter para valores reais em USD usando o scaler
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_real = scaler.inverse_transform(y_pred)

    mae_real = mean_absolute_error(y_test_real, y_pred_real)
    rmse_real = np.sqrt(mean_squared_error(y_test_real, y_pred_real))

    print(f"\nMetricas (valores reais em USD):")
    print(f"  MAE:  ${mae_real:.2f}")
    print(f"  RMSE: ${rmse_real:.2f}")

    # PASSO 5: Calcular R² (quanto da variancia o modelo explica)
    r_squared = r2_score(y_test, y_pred)
    r_squared_real = r2_score(y_test_real, y_pred_real)

    print(f"\nQualidade do ajuste:")
    print(f"  R² (normalizado): {r_squared:.4f}")
    print(f"  R² (real):        {r_squared_real:.4f}")
    print(f"  Interpretacao: Modelo explica {r_squared_real*100:.1f}% da variancia")

    # PASSO 6: Calcular Directional Accuracy (acerta a direcao?)
    # Compara se a direcao (sobe/desce) foi acertada
    y_test_direction = np.diff(y_test_real) > 0
    y_pred_direction = np.diff(y_pred_real.flatten()) > 0
    directional_accuracy = np.mean(y_test_direction == y_pred_direction)

    print(f"\nAccuracy Direcional (crucial em trading):")
    print(f"  Acerta direcao: {directional_accuracy*100:.1f}%")
    print(f"  Interpretacao: Prevê corretamente sobe/desce em {directional_accuracy*100:.1f}% dos casos")

    # PASSO 7: Salvar resultados profissionais em JSON
    metrics = {
        'symbol': SYMBOL,
        'mae_normalized': float(mae),
        'rmse_normalized': float(rmse),
        'mape': float(mape),
        'mae_real': float(mae_real),
        'rmse_real': float(rmse_real),
        'r_squared': float(r_squared_real),
        'directional_accuracy': float(directional_accuracy),
        'n_test_samples': int(len(y_test))
    }

    all_metrics[SYMBOL] = metrics

    with open(f'{MODELS_DIR}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetricas salvas em {MODELS_DIR}/metrics.json")

# RESUMO FINAL
print("\n" + "=" * 60)
print("ETAPA 4 CONCLUIDA COM SUCESSO")
print("=" * 60)

if all_metrics:
    print(f"\nResumo de performance dos modelos:")
    print(f"\n{'Ticker':<10} {'MAPE':<12} {'R²':<10} {'Dir. Acc':<12} {'MAE (USD)':<12}")
    print("-" * 56)
    for symbol, metrics in all_metrics.items():
        mape_str = f"{metrics['mape']:.2f}%"
        r2_str = f"{metrics['r_squared']:.3f}"
        dir_acc_str = f"{metrics['directional_accuracy']*100:.1f}%"
        mae_str = f"${metrics['mae_real']:.2f}"
        print(f"{symbol:<10} {mape_str:<12} {r2_str:<10} {dir_acc_str:<12} {mae_str:<12}")
