"""
Script para predecir la fiabilidad del modelo de Regresión Logística Bayesiana.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Añadir el directorio raíz al path para importar módulos
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.models.bayesian_logistics import BayesianLogisticRegression
from src.utils.io import load_data
from src.utils.preprocesing import fit_scaler, apply_scaler
from src.utils.split import split_data, split_col

# --- Global Parameters -------------------------------
SEED = 42
N_SAMPLES = 6000
OUTPUT_DIR = root_dir / "artifacts"
# -----------------------------------------------------

def calculate_predictive_entropy(probs):
    """
    Calcula la entropía predictiva para cada muestra.
    
    H(y|x) = -(p * log(p) + (1-p) * log(1-p))
    
    Donde p es la probabilidad de la clase positiva.
    La entropía es máxima (log(2) ≈ 0.693) cuando p = 0.5 (máxima incertidumbre).
    La entropía es mínima (0) cuando p = 0 o p = 1 (máxima certeza).
    
    Parámetros:
    ----------
    probs : np.ndarray
        Probabilidades de la clase positiva.
    
    Retorna:
    -------
    np.ndarray : Entropía predictiva para cada muestra.
    """
    p = np.clip(probs, 1e-10, 1 - 1e-10)
    entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return entropy

def calculate_confidence(probs):
    """
    Calcula la confianza del modelo para cada predicción.
    
    Confianza = max(p, 1-p)
    
    Parámetros:
    ----------
    probs : np.ndarray
        Probabilidades de la clase positiva.
    
    Retorna:
    -------
    np.ndarray : Confianza para cada muestra (entre 0.5 y 1.0).
    """
    return np.maximum(probs, 1 - probs)

def main():
    print("="*60)
    print("ANÁLISIS DE FIABILIDAD DEL MODELO BAYESIANO")
    print("="*60)
    
    # 1. Cargar datos
    print("\n[1/6] Cargando datos...")
    df = load_data()
    print(f"   Dataset cargado: {len(df)} muestras")
    
    # 2. Dividir en conjuntos de entrenamiento, validación y prueba
    print("\n[2/6] Dividiendo datos...")
    df_train, df_val, df_test = split_data(df, test_size=0.2, val_ratio=0.5, seed=SEED)
    print(f"   Train: {len(df_train)} muestras")
    print(f"   Val:   {len(df_val)} muestras")
    print(f"   Test:  {len(df_test)} muestras")
    
    # 3. Separar características (X) y etiquetas (y)
    X_train, y_train = split_col(df_train, col_name="DEATH_EVENT")
    X_test, y_test = split_col(df_test, col_name="DEATH_EVENT")
    
    # 4. Preprocesamiento: Normalización de datos
    print("\n[3/6] Normalizando datos...")
    scaler, X_train_norm = fit_scaler(X_train=X_train)
    X_test_norm = apply_scaler(scaler=scaler, X=X_test)
    
    # 5. Entrenamiento del modelo
    print(f"\n[4/6] Entrenando modelo Bayesiano con {N_SAMPLES} muestras MCMC...")
    model = BayesianLogisticRegression(
        num_samples=N_SAMPLES, 
        step_size=0.02, 
        burn_in=0.1, 
        seed=SEED, 
        init="random"
    )
    model.fit(X_train_norm, y_train)
    print("   Modelo entrenado correctamente")
    
    # 6. Predicción y análisis de fiabilidad
    print("\n[5/6] Calculando predicciones y métricas de fiabilidad...")
    probs = model.predict_proba(X_test_norm)
    preds = model.predict(X_test_norm)
    
    # Calcular métricas de incertidumbre
    entropy = calculate_predictive_entropy(probs)
    confidence = calculate_confidence(probs)
    
    # Calcular precisión
    accuracy = np.mean(preds == y_test)
    
    print(f"   Accuracy en test: {accuracy:.4f}")
    print(f"   Entropía media: {entropy.mean():.4f} (máximo posible: {np.log(2):.4f})")
    print(f"   Confianza media: {confidence.mean():.4f}")
    
    # 7. Análisis detallado
    print("\n[6/6] Generando reporte de fiabilidad...")
    
    # Identificar las muestras más inciertas (mayor entropía)
    most_uncertain_idx = np.argsort(entropy)[-10:][::-1]
    
    # Identificar las muestras más confiables (menor entropía)
    most_confident_idx = np.argsort(entropy)[:10]
    
    print("\n" + "="*60)
    print("RESUMEN DE FIABILIDAD")
    print("="*60)
    
    print(f"\nNúmero de muestras analizadas: {len(y_test)}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"\nEstadísticas de Entropía Predictiva:")
    print(f"  - Media:    {entropy.mean():.4f}")
    print(f"  - Std Dev:  {entropy.std():.4f}")
    print(f"  - Min:      {entropy.min():.4f}")
    print(f"  - Max:      {entropy.max():.4f}")
    print(f"  - Mediana:  {np.median(entropy):.4f}")
    
    print(f"\nEstadísticas de Confianza:")
    print(f"  - Media:    {confidence.mean():.4f}")
    print(f"  - Std Dev:  {confidence.std():.4f}")
    print(f"  - Min:      {confidence.min():.4f}")
    print(f"  - Max:      {confidence.max():.4f}")
    
    # Distribución de confianza
    high_conf = np.sum(confidence >= 0.9)
    medium_conf = np.sum((confidence >= 0.7) & (confidence < 0.9))
    low_conf = np.sum(confidence < 0.7)
    
    print(f"\nDistribución de Confianza:")
    print(f"  - Alta (≥0.9):        {high_conf} muestras ({high_conf/len(y_test)*100:.1f}%)")
    print(f"  - Media [0.7-0.9):    {medium_conf} muestras ({medium_conf/len(y_test)*100:.1f}%)")
    print(f"  - Baja (<0.7):        {low_conf} muestras ({low_conf/len(y_test)*100:.1f}%)")
    
    print("\n" + "-"*60)
    print("TOP 10 MUESTRAS MÁS INCIERTAS (Alta Entropía)")
    print("-"*60)
    print(f"{'Índice':<8} {'P(y=1)':<10} {'Entropía':<12} {'Confianza':<12} {'Real':<8} {'Pred':<8}")
    print("-"*60)
    for idx in most_uncertain_idx:
        print(f"{idx:<8} {probs[idx]:<10.4f} {entropy[idx]:<12.4f} {confidence[idx]:<12.4f} {y_test[idx]:<8} {preds[idx]:<8}")
    
    print("\n" + "-"*60)
    print("TOP 10 MUESTRAS MÁS CONFIABLES (Baja Entropía)")
    print("-"*60)
    print(f"{'Índice':<8} {'P(y=1)':<10} {'Entropía':<12} {'Confianza':<12} {'Real':<8} {'Pred':<8}")
    print("-"*60)
    for idx in most_confident_idx:
        print(f"{idx:<8} {probs[idx]:<10.4f} {entropy[idx]:<12.4f} {confidence[idx]:<12.4f} {y_test[idx]:<8} {preds[idx]:<8}")
    
    # 8. Guardar resultados en CSV
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "reliability_predictions.csv"
    
    results_df = pd.DataFrame({
        'probabilidad': probs,
        'prediccion': preds,
        'etiqueta_real': y_test,
        'entropia': entropy,
        'confianza': confidence,
        'correcto': (preds == y_test).astype(int)
    })
    
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Resultados guardados en: {output_path}")
    print("\n" + "="*60)
    print("ANÁLISIS COMPLETADO")
    print("="*60)

if __name__ == "__main__":
    main()
