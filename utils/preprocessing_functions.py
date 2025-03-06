import numpy as np
import pandas as pd


def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Divide los datos en conjuntos de entrenamiento y prueba manteniendo la
    proporción de clases (stratify).
    
    Parámetros:
    -----------
    X : pandas.DataFrame
        Características
    y : pandas.Series
        Etiquetas
    test_size : float, default=0.2
        Proporción del conjunto de prueba (entre 0 y 1)
    random_state : int, default=None
        Semilla para reproducibilidad
        
    Retorna:
    --------
    X_train, X_test, y_train, y_test : Conjuntos de entrenamiento y prueba
    """
    # Establecer semilla para reproducibilidad
    if random_state is not None:
        np.random.seed(random_state)
    
    # Convertir los DataFrames a arrays NumPy
    X_data = X.values
    y_data = y.values
    
    # Obtener clases únicas y sus índices
    classes = np.unique(y_data)
    test_indices = []
    
    # Seleccionar índices para test manteniendo la estratificación
    for c in classes:
        # Encontrar índices de esta clase
        indices = np.where(y_data == c)[0]
        
        # Mezclar índices para aleatorizar
        np.random.shuffle(indices)
        
        # Seleccionar una proporción para test
        n_class_test = int(len(indices) * test_size)
        test_indices.extend(indices[:n_class_test])
    
    # Crear máscara booleana para dividir los datos
    mask = np.zeros(len(X), dtype=bool)
    mask[test_indices] = True
    
    # Dividir los datos
    X_train = pd.DataFrame(X_data[~mask], columns=X.columns)
    X_test = pd.DataFrame(X_data[mask], columns=X.columns)
    y_train = pd.Series(y_data[~mask], name=y.name)
    y_test = pd.Series(y_data[mask], name=y.name)
    
    return X_train, X_test, y_train, y_test



