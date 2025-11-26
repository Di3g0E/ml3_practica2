import pandas as pd
from pathlib import Path


def load_data(filename="fallo_cardiaco.csv"):
    """
    Carga un archivo CSV desde la carpeta 'data' ubicada en el directorio raíz del proyecto.

    Parámetros:
    ----------
    filename : str, opcional

    Retorna:
    -------
    pd.DataFrame
    """
    # Ruta relativa desde main.py
    data_path = Path(__file__).resolve().parent.parent.parent / "data" / filename
    
    if not data_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {data_path}")
    
    return pd.read_csv(data_path)
