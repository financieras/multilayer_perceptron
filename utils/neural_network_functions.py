# utils/neural_network_functions.py

import numpy as np
import json # Importamos la librería json
# ... (resto de imports y funciones: denseLayer, createNetwork, fit, forward_propagation, backward_propagation, actualizar_pesos, calcular_loss, etc.)


def guardar_modelo_json(network, filename):
    """
    Guarda la topología y los pesos de la red neuronal en un archivo JSON.

    Parámetros:
    -----------
    network : list
        Lista que representa la red neuronal, donde cada elemento es una capa.
    filename : str
        Ruta al archivo JSON donde se guardará el modelo.
    """
    model_config = []
    for layer in network:
        layer_config = {
            'input_shape': layer['input_shape'],
            'output_shape': layer['output_shape'],
            'activation': layer['activation'].__name__, # Guardamos el nombre de la función de activación
            'weights_initializer': layer['weights_initializer'].__name__, # Guardamos el nombre del inicializador
            'weights': layer['weights'].tolist(), # Convertimos numpy array a lista para JSON
            'biases': layer['biases'].tolist()   # Convertimos numpy array a lista para JSON
        }
        model_config.append(layer_config)

    with open(filename, 'w') as f:
        json.dump(model_config, f, indent=4) # Indentación para que el JSON sea legible


def cargar_modelo_json(filename):
    """
    Carga una red neuronal desde un archivo JSON.

    Parámetros:
    -----------
    filename : str
        Ruta al archivo JSON del modelo.

    Retorna:
    --------
    network : list
        Red neuronal cargada.
    """
    with open(filename, 'r') as f:
        model_config = json.load(f)

    network = []
    for layer_config in model_config:
        layer = denseLayer(
            layer_config['input_shape'],
            layer_config['output_shape'],
            activation=globals()[layer_config['activation']], # Obtenemos la función de activación por su nombre
            weights_initializer=globals()[layer_config['weights_initializer']] # Obtenemos el inicializador por su nombre
        )
        layer['weights'] = np.array(layer_config['weights']) # Convertimos lista a numpy array
        layer['biases'] = np.array(layer_config['biases'])   # Convertimos lista a numpy array
        network.append(layer)
    return createNetwork(network) # Reconstruimos la red utilizando createNetwork para mantener la estructura


def predict(network, X):
    """
    Realiza predicciones con la red neuronal para los datos de entrada X.

    Parámetros:
    -----------
    network : list
        Red neuronal entrenada.
    X : numpy.ndarray
        Datos de entrada para la predicción.

    Retorna:
    --------
    predictions : numpy.ndarray
        Predicciones del modelo (probabilidades).
    """
    output = forward_propagation(network, X)
    # Como es clasificación binaria con sigmoide, aplicamos un umbral de 0.5 para clasificar
    predicted_classes = (output > 0.5).astype(int)
    return predicted_classes # Retornamos las clases predichas (0 o 1)