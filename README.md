# Práctica 2: Regresión Logística Bayesiana

Este proyecto implementa un modelo de Regresión Logística Bayesiana utilizando el algoritmo de Metropolis-Hastings para la predicción de mortalidad en pacientes con insuficiencia cardíaca.

## Estructura del Proyecto

- `src/`: Código fuente del modelo y utilidades.
- `data/`: Conjunto de datos (`fallo_cardiaco.csv`).
- `models/`: Modelos entrenados guardados.
- `docs/`: Documentación y memoria del proyecto.
- `train.py`: Script de entrenamiento.
- `test.py`: Script de inferencia/test.

## Requisitos

Instalar las dependencias necesarias:

```bash
pip install -r requirements.txt
```

## Uso

### Entrenamiento

Para entrenar el modelo con los parámetros por defecto:

```bash
python train.py
```

Esto guardará el modelo entrenado en `models/model.joblib`.

### Inferencia

Para evaluar el modelo en el conjunto de prueba:

```bash
python test.py
```

## Detalles del Modelo

El modelo utiliza inferencia Bayesiana para estimar la distribución posterior de los pesos de la regresión logística. Esto permite no solo predecir la clase, sino también estimar la incertidumbre asociada a cada predicción y la importancia de cada característica.