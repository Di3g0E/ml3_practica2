from src.models.bayesian_logistics import BayesianLogisticRegression
from src.utils.io import load_data
from src.utils.preprocesing import fit_scaler, apply_scaler
from src.utils.split import split_data, split_col

# --- Global Parameters -------------------------------
SEED = 42
N_SAMPLES = 6000
# -----------------------------------------------------

# 1. Cargar datos
print("Cargando datos...")
df = load_data()

# 2. Dividir en conjuntos de entrenamiento, validación y prueba
print("Dividiendo datos...")
df_train, df_val, df_test = split_data(df, test_size=0.2, val_ratio=0.5, seed=SEED)

# 3. Separar características (X) y etiquetas (y)
# Identificamos 'DEATH_EVENT' como la columna objetivo
X_train, y_train = split_col(df_train, col_name="DEATH_EVENT")
X_val, y_val = split_col(df_val, col_name="DEATH_EVENT")

# 4. Preprocesamiento: Normalización de datos
# Es crucial normalizar para que MCMC converja mejor y los coeficientes sean comparables
print("Normalizando datos...")
scaler, X_train_norm = fit_scaler(X_train=X_train)
X_val_norm = apply_scaler(scaler=scaler, X=X_val)

# 5. Definición y entrenamiento de modelo de Regresión Logística Bayesiana
print(f"Entrenando modelo Bayesiano con {N_SAMPLES} muestras...")
model = BayesianLogisticRegression(num_samples=N_SAMPLES, step_size=0.02, burn_in=0.1, seed=SEED)
model.fit(X_train_norm, y_train)

# 6. Predicción sobre el conjunto de validación
probs = model.predict_proba(X_val_norm)
preds = model.predict(X_val_norm)
print("Predicciones completadas.")

# 7. Análisis de incertidumbre
# Obtenemos la media y desviación estándar de los coeficientes aprendidos
summary = model.summary()
print("\n--- Resumen del Modelo ---")
print("Media de coeficientes (Influencia):", summary["mean"])
print("Desviación estándar (Incertidumbre):", summary["std"])
