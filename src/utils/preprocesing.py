from sklearn.preprocessing import StandardScaler

def fit_scaler(X_train):
    """
    Ajusta un StandardScaler solo con el conjunto de entrenamiento.

    Parámetros:
    ----------
    X_train : Matriz de características del conjunto de entrenamiento.

    Retorna:
    -------
    scaler : Objeto ajustado que puede usarse para transformar otros conjuntos.
    X_train_scaled : Conjunto de entrenamiento normalizado.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    return scaler, X_train_scaled

def apply_scaler(scaler, X):
    """
    Aplica un StandardScaler previamente ajustado a un nuevo conjunto.

    Parámetros:
    ----------
    scaler : Objeto previamente ajustado con fit_scaler().
    X : Matriz de características a transformar.

    Retorna:
    -------
    X_scaled : Matriz transformada.
    """
    return scaler.transform(X)
