# utils/evaluation_functions.py

import numpy as np

def accuracy(y_true, y_pred):
    """
    Calcula la accuracy (exactitud) de las predicciones.

    Parámetros:
    -----------
    y_true : numpy.ndarray
        Etiquetas verdaderas (valores reales).
    y_pred : numpy.ndarray
        Etiquetas predichas.

    Retorna:
    --------
    accuracy : float
        Accuracy del modelo.
    """
    y_true = np.array(y_true).flatten() # Aseguramos que sean arrays 1D
    y_pred = np.array(y_pred).flatten()
    return np.mean(y_true == y_pred)