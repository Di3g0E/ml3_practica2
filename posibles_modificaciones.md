# Posibles Modificaciones para Defensa de Práctica

Lista de retos probables, ordenados por dificultad, junto con pistas de cómo resolverlos.

## Nivel 1: Modificaciones Simples (5-10 mins)

### 1. Cambiar la inicialización de los pesos
**Reto:** "El modelo actual permite inicialización aleatoria o a ceros. Quiero que añadas una opción para inicializar los pesos con un valor fijo, por ejemplo, todos a 0.5."
**Solución:**
- Ve a `src/models/bayesian_logistics.py`.
- En `__init__`, añade un argumento o simplemente modifica la lógica en `fit`.
- En el método `fit`, busca el bloque `if self.init == "zero":`.
- Añade `elif self.init == "fixed": current_weights = np.full(n_features, 0.5)`.

### 2. Modificar el umbral de decisión en test
**Reto:** "Por defecto clasificamos como muerte si la probabilidad es > 0.5. Dado que es un problema médico, queremos ser más conservadores. Cambia el script de test para que el usuario pueda pasar el umbral por consola."
**Solución:**
- Ve a `test.py`.
- En `parse_args`, añade `parser.add_argument("--threshold", type=float, default=0.5)`.
- En `main`, cambia `preds = model.predict(X_test_norm)` por `preds = model.predict(X_test_norm, threshold=args.threshold)`.

### 3. Cambiar el número de iteraciones de Burn-in
**Reto:** "Quiero que el burn-in no sea un porcentaje, sino un número fijo de muestras que yo decida."
**Solución:**
- Ve a `src/models/bayesian_logistics.py`.
- En `__init__`, simplifica la lógica de `self.burn_in` para que acepte directamente el entero si se pasa como tal, o modifica `train.py` para pasar un entero.

## Nivel 2: Modificaciones Medias (15-25 mins)

### 4. Añadir una métrica de evaluación extra: F1-Score
**Reto:** "La exactitud (accuracy) no es suficiente. Implementa manualmente el cálculo del F1-Score en `test.py`."
**Solución:**
- Necesitas calcular True Positives (TP), False Positives (FP) y False Negatives (FN).
- En `test.py`:
  ```python
  tp = ((preds == 1) & (y_test == 1)).sum()
  fp = ((preds == 1) & (y_test == 0)).sum()
  fn = ((preds == 0) & (y_test == 1)).sum()
  precision = tp / (tp + fp) if (tp + fp) > 0 else 0
  recall = tp / (tp + fn) if (tp + fn) > 0 else 0
  f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
  print(f"F1-Score: {f1:.4f}")
  ```

### 5. Implementar un "Learning Rate Decay" en MCMC
**Reto:** "El `step_size` es fijo. Haz que disminuya con el tiempo para explorar menos al final."
**Solución:**
- Ve a `src/models/bayesian_logistics.py`, método `fit`.
- Dentro del bucle `for _ in range(self.num_samples):`, modifica `self.step_size`.
- Ejemplo: `current_step_size = self.step_size * (1 - (_ / self.num_samples))`.
- Usa `current_step_size` en la generación de `proposal`.

### 6. Guardar el historial de Loss (Log-Likelihood)
**Reto:** "Quiero ver cómo evoluciona la verosimilitud (log-likelihood) durante el entrenamiento. Guárdala y devuélvela."
**Solución:**
- En `src/models/bayesian_logistics.py`, añade `self.likelihood_history_ = []` en `__init__`.
- En `fit`, dentro del bucle, añade `self.likelihood_history_.append(current_ll)`.
- En `train.py`, después de entrenar, puedes imprimir los primeros/últimos valores o guardarlos.

## Nivel 3: Modificaciones Avanzadas (30-45 mins)

### 7. Validación Cruzada (Cross-Validation)
**Reto:** "En lugar de un solo split train/test, implementa K-Fold Cross Validation en `train.py`."
**Solución:**
- Esto requiere cambiar el flujo en `train.py`.
- Tendrías que hacer un bucle (ej. 5 veces), dividir los datos de forma diferente cada vez (usando índices o `sklearn.model_selection.KFold` si te dejan, o manual con numpy), entrenar, evaluar y promediar resultados.
- *Nota:* Si te piden esto sin librerías externas como sklearn, es complejo para 45 mins, pero conceptualmente debes saber explicarlo.

### 8. Regularización (Prior)
**Reto:** "El modelo actual asume un prior plano (uniforme). Añade un prior Gaussiano a los pesos (L2 regularization) en el cálculo de aceptación."
**Solución:**
- Modifica `_log_likelihood` o crea `_log_posterior`.
- `log_posterior = log_likelihood + log_prior`.
- Si el prior es Gaussiano $N(0, \sigma^2)$, `log_prior` es proporcional a $-\sum w_i^2 / (2\sigma^2)$.
- Añade este término al `log_ratio` en el paso de Metropolis-Hastings.

## Consejos Generales para la Defensa
1.  **Conoce tu código:** Revisa línea por línea `bayesian_logistics.py`. Debes saber qué hace `_sigmoid`, por qué sumamos logaritmos en `_log_likelihood` (para evitar underflow), y cómo funciona el criterio de aceptación `np.exp(log_ratio)`.
2.  **Usa `print`:** Si te atascas depurando, pon `print` dentro de los bucles para ver qué valores toman las variables.
3.  **Argumentos:** Si te piden cambiar un hiperparámetro, lo más elegante es añadirlo a `argparse` en `train.py` o `test.py`, no "hardcodearlo".
