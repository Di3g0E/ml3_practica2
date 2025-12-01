import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.models.bayesian_logistics import BayesianLogisticRegression
from src.utils.io import load_data
from src.utils.preprocesing import fit_scaler, apply_scaler
from src.utils.split import split_data, split_col

def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)

    # Probabilidades y predicciones
    probas = model.predict_proba(X_test)
    preds   = model.predict(X_test)

    # Métricas
    acc   = accuracy_score(y_test, preds)
    ll    = -log_loss(y_test, probas)          
    auc   = roc_auc_score(y_test, probas)

    # Información del posterior (media y std de los coeficientes)
    summary = model.summary()
    return {
        "accuracy": acc,
        "log_likelihood": ll,
        "roc_auc": auc,
        "coef_mean": summary["mean"],
        "coef_std":  summary["std"]
    }

def run_experiment(init_type, X_train, y_train, X_test, y_test, n_runs=10):
    results = []
    print(f"Running experiment for init='{init_type}' with {n_runs} runs...")
    for i in range(n_runs):
        # Cambiamos la semilla para cada repetición
        seed = 42 + i
        model = BayesianLogisticRegression(
            num_samples=6000,
            step_size=0.02,
            burn_in=0.1,
            seed=seed,
            init=init_type,
        )
        res = evaluate_model(model, X_train, y_train, X_test, y_test)
        results.append(res)
    return results

def summarize(results):
    keys = ["accuracy", "log_likelihood", "roc_auc"]
    summary = {}
    for k in keys:
        vals = np.array([r[k] for r in results])
        summary[k] = {"mean": vals.mean(), "std": vals.std()}
    return summary

def main():
    # 1. Cargar y preparar datos (Igual que main.py)
    print("Cargando y procesando datos...")
    df = load_data()
    
    SEED = 42
    df_train, df_val, df_test = split_data(df, test_size=0.2, val_ratio=0.5, seed=SEED)
    
    X_train, y_train = split_col(df_train, col_name="DEATH_EVENT")
    X_val, y_val = split_col(df_val, col_name="DEATH_EVENT")
    
    scaler, X_train_norm = fit_scaler(X_train=X_train)
    X_val_norm = apply_scaler(scaler=scaler, X=X_val)

    # 2. Ejecutar experimentos
    zero_res   = run_experiment("zero", X_train_norm, y_train, X_val_norm, y_val, n_runs=10)
    random_res = run_experiment("random", X_train_norm, y_train, X_val_norm, y_val, n_runs=10)

    # 3. Mostrar resultados
    print("\n=== Inicialización a CERO ===")
    s_zero = summarize(zero_res)
    for k, v in s_zero.items():
        print(f"{k}: {v['mean']:.4f} +/- {v['std']:.4f}")

    print("\n=== Inicialización ALEATORIA ===")
    s_random = summarize(random_res)
    for k, v in s_random.items():
        print(f"{k}: {v['mean']:.4f} +/- {v['std']:.4f}")

if __name__ == "__main__":
    main()
