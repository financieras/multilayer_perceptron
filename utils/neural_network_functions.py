## utils/neural_network_functions.py
import numpy as np
import json

# --- Activation Functions ---
def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of the sigmoid function."""
    return sigmoid(x) * (1 - sigmoid(x))


def denseLayer(input_shape, output_shape, activation, weights_initializer):
    """
    Creates a dense layer for a neural network.

    Args:
        input_shape (int): Number of input features.
        output_shape (int): Number of neurons in this layer.
        activation (function): Activation function for this layer.
        weights_initializer (function): Weight initialization function.

    Returns:
        dict: A dictionary representing the dense layer.
    """
    layer = {}
    layer['input_shape'] = input_shape
    layer['output_shape'] = output_shape
    layer['activation'] = activation
    layer['weights_initializer'] = weights_initializer
    layer['weights'] = weights_initializer((input_shape, output_shape)) # Weights matrix (output_shape x input_shape)
    layer['biases'] = np.zeros((1, output_shape)) # Bias vector (1 x output_shape)
    return layer


def createNetwork(layers):
    """
    Creates a neural network from a list of layers.

    Args:
        layers (list): A list of layer dictionaries, defining the network architecture.

    Returns:
        list: The neural network, which is simply the list of layers.
    """
    network = layers # For now, just returns the list of layers as the network
    return network


def save_model_json(network, filename):
    """
    Saves the neural network topology and weights to a JSON file.

    Args:
        network (list): List representing the neural network, where each element is a layer.
        filename (str): Path to the JSON file where the model will be saved.
    """
    model_config = []
    for layer in network:
        layer_config = {
            'input_shape': layer['input_shape'],
            'output_shape': layer['output_shape'],
            'activation': layer['activation'].__name__,  # Save activation function name
            'weights_initializer': layer['weights_initializer'].__name__,  # Save weights initializer name
            'weights': layer['weights'].tolist(),  # Convert numpy array to list for JSON
            'biases': layer['biases'].tolist()   # Convert numpy array to list for JSON
        }
        model_config.append(layer_config)

    with open(filename, 'w') as f:
        json.dump(model_config, f, indent=4)  # Indentation for readable JSON


def load_model_json(filename):
    """
    Loads a neural network from a JSON file.

    Args:
        filename (str): Path to the JSON model file.

    Returns:
        list: Loaded neural network.
    """
    with open(filename, 'r') as f:
        model_config = json.load(f)

    network = []
    for layer_config in model_config:
        layer = denseLayer(
            layer_config['input_shape'],
            layer_config['output_shape'],
            activation=globals()[layer_config['activation']],  # Get activation function by name
            weights_initializer=globals()[layer_config['weights_initializer']]  # Get initializer by name
        )
        layer['weights'] = np.array(layer_config['weights'])  # Convert list to numpy array
        layer['biases'] = np.array(layer_config['biases'])   # Convert list to numpy array
        network.append(layer)
    return createNetwork(network)  # Reconstruct network using createNetwork to maintain structure


def predict(network, X):
    """
    Performs predictions with the neural network for input data X.

    Args:
        network (list): Trained neural network.
        X (numpy.ndarray): Input data for prediction.

    Returns:
        numpy.ndarray: Model predictions (probabilities).
    """
    output = forward_propagation(network, X)
    # For binary classification with sigmoid, apply a threshold of 0.5 to classify
    predicted_classes = (output > 0.5).astype(int)
    return predicted_classes  # Returns predicted classes (0 or 1)




def forward_propagation(network, X):
    """
    Realiza la propagación hacia adelante a través de la red neuronal.

    Args:
        network (list): La red neuronal (lista de capas).
        X (numpy.ndarray): Datos de entrada (batch de ejemplos).

    Returns:
        numpy.ndarray: Salida de la red neuronal (predicciones).
    """
    input_layer = X # La entrada a la primera capa es X
    for layer in network: # Iteramos sobre cada capa de la red
        weights = layer['weights'] # Obtenemos los pesos de la capa actual
        biases = layer['biases']   # Obtenemos los biases de la capa actual
        activation_function = layer['activation'] # Obtenemos la función de activación

        # Cálculo de la combinación lineal (z = W*input + b)
        z = np.dot(input_layer, weights) + biases # Producto punto de pesos y la entrada de la capa anterior, sumamos el bias
        z = z

        # Aplicación de la función de activación (a = activation(z))
        output_layer = activation_function(z) # Aplicamos la función de activación al resultado de la combinación lineal

        input_layer = output_layer # La salida de la capa actual se convierte en la entrada para la siguiente capa

    return output_layer # La salida de la última capa es la predicción final de la red



def fit(network, X_train, y_train, X_valid, y_valid, loss_function="binary_crossentropy", learning_rate=0.01, batch_size=32, epochs=100):
    """
    Entrena la red neuronal utilizando los datos de entrenamiento y validación.

    Args:
        network (list): La red neuronal a entrenar.
        X_train (numpy.ndarray): Datos de entrenamiento (características).
        y_train (numpy.ndarray): Etiquetas de entrenamiento.
        X_valid (numpy.ndarray): Datos de validación (características).
        y_valid (numpy.ndarray): Etiquetas de validación.
        loss (str, optional): Función de pérdida a utilizar. Defaults to "binary_crossentropy".
        learning_rate (float, optional): Tasa de aprendizaje. Defaults to 0.01.
        batch_size (int, optional): Tamaño del batch para el descenso de gradiente. Defaults to 32.
        epochs (int, optional): Número de épocas de entrenamiento. Defaults to 100.

    Returns:
        dict: Diccionario que contiene el historial del entrenamiento (loss y accuracy en entrenamiento y validación).
    """
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []} # Diccionario para guardar el historial de entrenamiento
    n_samples = X_train.shape[0] # Número total de ejemplos de entrenamiento

    for epoch in range(epochs): # Iterar sobre el número de épocas
        # --- Mini-batch Gradient Descent ---
        # Crear batches de datos de entrenamiento para cada época
        indices = np.random.permutation(n_samples) # Permutación aleatoria de los índices de entrenamiento
        X_train_shuffled = X_train.iloc[indices] # Reordenar X_train usando los índices permutados
        y_train_shuffled = y_train[indices]    # Reordenar y_train usando los mismos índices

        for start_index in range(0, n_samples, batch_size): # Iterar sobre los batches
            end_index = min(start_index + batch_size, n_samples) # Calcular el índice final del batch
            batch_X = X_train_shuffled[start_index:end_index] # Extraer el batch de características
            batch_y = y_train_shuffled[start_index:end_index] # Extraer el batch de etiquetas

            # --- Forward Propagation ---
            # Almacenar las salidas ANTES de la activación en cada capa para backpropagation
            output_batch = batch_X # Inicializar la entrada para la primera capa con el batch de entrada
            for layer in network:
                layer['output_before_activation'] = np.dot(output_batch, layer['weights']) + layer['biases'] # Calcular z = W*input + b
                layer['output'] = layer['activation'](layer['output_before_activation']) # Calcular a = activation(z)
                output_batch = layer['output'] # La salida de esta capa es la entrada para la siguiente

            predictions_batch = output_batch # Las predicciones para el batch actual son la salida de la última capa

            # --- Calcular Loss ---
            loss_batch = calculate_loss(predictions_batch, batch_y, loss_function) # Calcular la loss para el batch actual

            # --- Backward Propagation ---
            gradients_batch = backward_propagation(network, predictions_batch, batch_y, batch_X, loss_function) # Calcular los gradientes para el batch actual

            # --- Update Weights ---
            network = update_weights(network, gradients_batch, learning_rate) # Actualizar los pesos de la red usando los gradientes

        # --- Calcular métricas después de cada época ---
        # Loss y Accuracy en el conjunto de entrenamiento
        predictions_train = predict(network, X_train) # Predicciones sobre todo el conjunto de entrenamiento
        loss_train = calculate_loss(predictions_train, y_train, loss_function) # Calcular la loss en entrenamiento
        accuracy_train = calculate_accuracy(predictions_train, y_train) # Calcular accuracy en entrenamiento

        # Loss y Accuracy en el conjunto de validación
        predictions_valid = predict(network, X_valid) # Predicciones sobre el conjunto de validación
        loss_valid = calculate_loss(predictions_valid, y_valid, loss_function) # Calcular la loss en validación
        accuracy_valid = calculate_accuracy(predictions_valid, y_valid) # Calcular accuracy en validación

        # --- Registrar historial ---
        history['loss'].append(loss_train) # Guardar la loss de entrenamiento en el historial
        history['val_loss'].append(loss_valid) # Guardar la loss de validación en el historial
        history['accuracy'].append(accuracy_train) # Guardar la accuracy de entrenamiento
        history['val_accuracy'].append(accuracy_valid) # Guardar la accuracy de validación

        # --- Imprimir el progreso en cada época ---
        print(f"Epoch {epoch+1}/{epochs}, Loss (train/val): {loss_train:.4f}/{loss_valid:.4f}, Accuracy (train/val): {accuracy_train:.4f}/{accuracy_valid:.4f}")

    return history # Retornar el historial de entrenamiento



def backward_propagation(network, predictions, labels, X_train, loss_function="binary_crossentropy"):
    """
    Realiza la retropropagación para calcular los gradientes de la loss.

    Args:
        network (list): La red neuronal.
        predictions (numpy.ndarray): Predicciones del modelo (salida de forward_propagation).
        labels (numpy.ndarray): Etiquetas reales.
        loss_function (str, optional): Función de pérdida utilizada.
                                         Defaults to "binary_crossentropy".

    Returns:
        dict: Diccionario que contiene los gradientes para cada capa.
              Las claves del diccionario son 'layer_index' (índice de la capa)
              y los valores son diccionarios con claves 'weights_gradients' y 'biases_gradients'.
    """
    gradients = {} # Diccionario para guardar los gradientes de cada capa
    n_layers = len(network) # Número de capas en la red

    # --- Gradiente para la capa de salida ---
    # (derivada de la loss con respecto a la salida de la última capa)
    if loss_function == "binary_crossentropy":
        # Derivada de Binary Crossentropy con respecto a las predicciones (para sigmoide output)
        output_error_signal = -(labels - predictions) # Simplification for sigmoid + binary crossentropy derivative.
    else:
        raise ValueError(f"Backward propagation not implemented for loss function '{loss_function}'.")

    # Para la última capa (capa de salida)
    layer_index = n_layers - 1 # Índice de la última capa (empezando desde 0)
    layer = network[layer_index] # Obtenemos la última capa
    prev_layer_output = network[layer_index - 1]['output'] if layer_index > 0 else X_train # Salida de la capa anterior (o entrada X si es la primera capa oculta)

    # Gradiente de la salida de la capa con respecto a la combinación lineal z (derivada de la función de activación)
    activation_prime = sigmoid_derivative(layer['output_before_activation']) # Usamos la derivada de la sigmoide

    # Error signal para la capa de salida
    error_signal = output_error_signal * activation_prime
    gradients[layer_index] = {'error_signal': error_signal} # Store error signal para backpropagation a capas previas

    # Gradientes para los pesos y biases de la capa de salida
    weights_gradients = np.dot(prev_layer_output.T, error_signal) # Gradiente pesos = error_signal * output_capa_anterior(transpuesta)
    biases_gradients = np.sum(error_signal, axis=0, keepdims=True) # Gradiente biases = suma del error_signal sobre el batch

    gradients[layer_index].update({'weights_gradients': weights_gradients, 'biases_gradients': biases_gradients}) # Update layer gradients

    # --- Retropropagación para capas ocultas ---
    # Iterar sobre las capas ocultas en orden inverso (desde la penúltima hasta la primera)
    for layer_index in range(n_layers - 2, -1, -1): # Iteramos desde la penúltima capa hasta la primera (índices n_layers-2, n_layers-3, ..., 0)
        layer = network[layer_index] # Obtenemos la capa actual
        next_layer = network[layer_index + 1] # Obtenemos la capa siguiente (capa hacia adelante en la propagación)
        prev_layer_output = network[layer_index - 1]['output'] if layer_index > 0 else X_train # Salida de la capa anterior (o entrada X si es la primera capa oculta)

        # Gradiente de la salida de la capa actual con respecto a la combinación lineal z (derivada de la función de activación)
        activation_prime = sigmoid_derivative(layer['output_before_activation']) # Usamos la derivada de la sigmoide

        # Error signal para la capa actual (retropropagación del error de la capa siguiente)
        error_signal = np.dot(gradients[layer_index + 1]['error_signal'], next_layer['weights'].T) * activation_prime # error_signal = (error_signal_capa_siguiente * pesos_capa_siguiente) * derivada_activacion_capa_actual


        gradients[layer_index] = {'error_signal': error_signal} # Store error signal for this layer

        # Gradientes para los pesos y biases de la capa actual
        weights_gradients = np.dot(prev_layer_output.T, error_signal) # Gradiente pesos = error_signal * output_capa_anterior(transpuesta)
        biases_gradients = np.sum(error_signal, axis=0, keepdims=True) # Gradiente biases = suma del error_signal sobre el batch

        gradients[layer_index].update({'weights_gradients': weights_gradients, 'biases_gradients': biases_gradients}) # Update layer gradients


    return gradients # Retornamos el diccionario de gradientes



def update_weights(network, gradients, learning_rate):
    """
    Actualiza los pesos y biases de la red neuronal usando los gradientes calculados.

    Args:
        network (list): La red neuronal.
        gradients (dict): Diccionario de gradientes calculado por backward_propagation.
        learning_rate (float): Tasa de aprendizaje.

    Returns:
        list: Red neuronal actualizada.
    """
    updated_network = [] # Lista para construir la red neuronal actualizada
    n_layers = len(network) # Número de capas

    for layer_index in range(n_layers): # Iteramos sobre cada capa
        layer = network[layer_index] # Obtenemos la capa actual
        layer_gradients = gradients[layer_index] # Obtenemos los gradientes de la capa actual

        # Actualizar pesos y biases usando descenso de gradiente
        updated_weights = layer['weights'] - learning_rate * layer_gradients['weights_gradients'] # pesos_nuevos = pesos_viejos - learning_rate * gradientes_pesos
        updated_biases = layer['biases'] - learning_rate * layer_gradients['biases_gradients'] # biases_nuevos = biases_viejos - learning_rate * gradientes_biases

        # Crear una nueva capa actualizada (manteniendo la estructura de diccionario)
        updated_layer = layer.copy() # Copiamos la capa original para no modificarla directamente
        updated_layer['weights'] = updated_weights # Asignamos los pesos actualizados
        updated_layer['biases'] = updated_biases   # Asignamos los biases actualizados
        updated_network.append(updated_layer) # Añadimos la capa actualizada a la nueva red

    return updated_network # Retornamos la red neuronal con los pesos actualizados



def calculate_loss(predictions, labels, loss_function="binary_crossentropy"):
    """
    Calcula la función de pérdida (loss) entre las predicciones y las etiquetas reales.

    Args:
        predictions (numpy.ndarray): Predicciones del modelo.
        labels (numpy.ndarray): Etiquetas reales (valores objetivo).
        loss_function (str, optional): Nombre de la función de pérdida a usar.
                                         Por ahora, solo se implementa 'binary_crossentropy'.
                                         Defaults to "binary_crossentropy".

    Returns:
        float: Valor de la función de pérdida (pérdida promedio sobre el batch).
    """
    if loss_function == "binary_crossentropy":
        # Binary Crossentropy Loss
        epsilon = 1e-15 # Pequeño valor para evitar log(0)
        predictions_clipped = np.clip(predictions, epsilon, 1 - epsilon) # Acotamos las predicciones para evitar log(0) o log(1)
        loss = -np.mean(labels * np.log(predictions_clipped) + (1 - labels) * np.log(1 - predictions_clipped)) # Fórmula de Binary Crossentropy
        return loss
    else:
        raise ValueError(f"Loss function '{loss_function}' not implemented.")



def he_uniform(shape):
    """
    Inicialización de pesos He Uniform (también conocida como inicialización Kaiming Uniform).

    Args:
        shape (tuple): Forma de la matriz de pesos a inicializar (output_shape, input_shape).

    Returns:
        numpy.ndarray: Matriz de pesos inicializada.
    """
    fan_in = shape[1] # fan_in es el número de entradas a la capa (segunda dimensión de la forma)
    limit = np.sqrt(6 / fan_in) # Límite para la distribución uniforme (factor para He Uniform)
    weights = np.random.uniform(-limit, limit, size=shape) # Inicializar pesos con distribución uniforme entre -limit y limit
    return weights



def calculate_accuracy(predictions, labels):
    """
    Calculates the accuracy between predictions and true labels for binary classification.

    Args:
        predictions (numpy.ndarray): Model predictions (probabilities).
        labels (numpy.ndarray): True labels (0 or 1).

    Returns:
        float: Accuracy as a percentage.
    """
    # Threshold predictions to get class labels (0 or 1)
    predicted_classes = (predictions > 0.5).astype(int)
    # Calculate the number of correct predictions
    correct_predictions = np.sum(predicted_classes == labels)
    # Calculate accuracy as percentage
    accuracy = correct_predictions / len(labels)
    return accuracy