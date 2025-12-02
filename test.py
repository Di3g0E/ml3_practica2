import joblib
from src.utils.io import load_data
from src.utils.preprocesing import apply_scaler
from src.utils.split import split_data, split_col
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Realizar inferencia con el modelo de Regresión Logística Bayesiana")
    parser.add_argument("--model_path", type=str, default="models/model.joblib", help="Ruta del modelo guardado")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria para el split de datos")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load saved model and scaler
    try:
        data = joblib.load(args.model_path)
        model = data["model"]
        scaler = data["scaler"]
    except FileNotFoundError:
        print(f"Error: No se encontró el modelo en {args.model_path}. Ejecuta train.py primero.")
        sys.exit(1)

    print("Cargando datos de prueba...")
    df = load_data()
    _, _, df_test = split_data(df, test_size=0.2, val_ratio=0.5, seed=args.seed)
    
    X_test, y_test = split_col(df_test, col_name="DEATH_EVENT")
    X_test_norm = apply_scaler(scaler=scaler, X=X_test)
    
    print("Realizando predicciones...")
    probs = model.predict_proba(X_test_norm)
    preds = model.predict(X_test_norm)
    
    print("\nPredicciones en conjunto de prueba:")
    print(preds)
    
    # Metricas
    accuracy = (preds == y_test).mean()
    print(f"\nExactitud en test: {accuracy:.4f}")

if __name__ == "__main__":
    main()
