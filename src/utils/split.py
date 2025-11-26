from sklearn.model_selection import train_test_split


def split_data(df, test_size=0.2, val_ratio=0.5, seed=42):
    """
    Divide un DataFrame en train, val y test.
    
    Parámetros:
    - df: DataFrame original
    - test_size: proporción total para test + val (ej. 0.2)
    - val_ratio: proporción dentro del test_size que será validación (ej. 0.5)
    - random_state: semilla para reproducibilidad
    
    Retorna:
    - df_train, df_val, df_test
    """
    # División inicial: train vs (val + test)
    df_train, df_temp = train_test_split(df, test_size=test_size, random_state=seed)
    
    # División secundaria: val vs test dentro de df_temp
    val_size = val_ratio
    df_val, df_test = train_test_split(df_temp, test_size=1 - val_size, random_state=seed)
    
    return df_train, df_val, df_test

def split_col(df, col_name="DEATH_EVENT"):
    """
    Separa un DataFrame en matriz de características (X) y vector de etiquetas (y) para clasificación binaria.

    Parámetros:
    ----------
    df : DataFrame original que contiene tanto las variables predictoras como la etiqueta.
    tag_name : Nombre de la columna que representa la etiqueta binaria. Por defecto es 'DEATH_EVENT'.

    Retorna:
    -------
    X_train : Matriz con las variables predictoras (todas las columnas excepto la etiqueta).
    y_train : Vector con las etiquetas binarias (columna especificada por tag_name).
    """
    X_train = df.drop(columns=[col_name]).values
    y_train = df[col_name].values

    return X_train, y_train
