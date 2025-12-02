from src.models.bayesian_logistics import BayesianLogisticRegression
from src.utils.io import load_data
from src.utils.preprocesing import fit_scaler, apply_scaler
from src.utils.split import split_data, split_col
import joblib

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Entrenar modelo de Regresión Logística Bayesiana")
    parser.add_argument("--samples", type=int, default=6000, help="Número de muestras MCMC")
    parser.add_argument("--step_size", type=float, default=0.02, help="Tamaño de paso para Metropolis-Hastings")
    parser.add_argument("--burn_in", type=float, default=0.1, help="Proporción de burn-in")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument("--model_path", type=str, default="models/model.joblib", help="Ruta para guardar el modelo")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("Cargando datos...")
    df = load_data()
    print("Dividiendo datos...")
    df_train, df_val, df_test = split_data(df, test_size=0.2, val_ratio=0.5, seed=args.seed)
    X_train, y_train = split_col(df_train, col_name="DEATH_EVENT")
    X_val, y_val = split_col(df_val, col_name="DEATH_EVENT")
    
    print("Normalizando datos...")
    scaler, X_train_norm = fit_scaler(X_train=X_train)
    X_val_norm = apply_scaler(scaler=scaler, X=X_val)
    
    print(f"Entrenando modelo Bayesiano con {args.samples} muestras...")
    model = BayesianLogisticRegression(num_samples=args.samples, step_size=args.step_size, burn_in=args.burn_in, seed=args.seed, init="random")
    model.fit(X_train_norm, y_train)
    
    # Guardar el modelo y el scaler
    joblib.dump({"model": model, "scaler": scaler}, args.model_path)
    print(f"Modelo guardado en {args.model_path}")
    
    # Realizar predicciones
    probs = model.predict_proba(X_val_norm)
    preds = model.predict(X_val_norm)
    print("Predicciones de validación completadas.")
    
    summary = model.summary()
    print("\n--- Resumen del Modelo ---")
    print("Media de coeficientes (Influencia):", summary["mean"])
    print("Desviación estándar (Incertidumbre):", summary["std"])

if __name__ == "__main__":
    main()
