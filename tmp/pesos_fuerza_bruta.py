import numpy as np
import pandas as pd
np.random.seed()       # OJO La semilla se fija para hacer prueba, en producción quirar el número

def forward_propagation(X, weights, biases):
    # Número de capas en la red
    num_layers = len(weights)
    #print("num_layers:", num_layers)    # imprime 3 para dos capas ocultas, ya que hay 3 conjuntos de pesos (2 ocultas + 1 de salida)

    # Inicializar la activación de la capa de entrada
    activation = X
    
    # Realizar la propagación hacia adelante capa por capa
    for l in range(num_layers):
        # Calcular la entrada ponderada (z)
        #z = np.dot(activation, weights[l].T) + biases[l]
        z = activation @ weights[l].T + biases[l]
        
        # Aplicar la función de activación (a)
        activation = sigmoid(z)
    
    # La salida de la red neuronal es la activación de la última capa
    return activation

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def calculate_error(predictions, y):
    # Asegurarse de que las predicciones estén en el rango (0, 1)
    predictions = np.clip(predictions, 1e-15, 1 - 1e-15)

    # Calcular la entropía cruzada binaria manualmente
    epsilon = 1e-15  # Para evitar el logaritmo de cero
    error = -np.mean(y * np.log(predictions + epsilon) + (1 - y) * np.log(1 - predictions + epsilon))
    return error

if __name__ == "__main__":
    # Cargar los datos preprocesados
    data = pd.read_csv('data/preprocessed_data.csv')
    
    # Separar las características (X) y el objetivo (y)
    X = data.iloc[:, 2:].values  # Características (features)
    y = data['diagnosis'].values  # Objetivo (diagnóstico)
    
    # Definir la estructura de la red neuronal
    num_inputs = X.shape[1]
    num_outputs = 1
    layer_sizes = [num_inputs, 31, 17, 11, 7, 3, num_outputs]  # Neuronas de las DOS capas ocultas

    # Número máximo de iteraciones
    max_iter = 100_000_000
    error_min = float('inf')

    for i in range(max_iter):
        # Generar nuevos pesos y sesgos aleatoriamente
        new_weights = [np.random.randn(layer_sizes[i+1], layer_sizes[i]) for i in range(len(layer_sizes)-1)]
        new_biases = [np.random.randn(1, layer_sizes[i+1]) for i in range(len(layer_sizes)-1)]

        # Realizar la propagación hacia adelante con los nuevos pesos y sesgos
        new_predictions = forward_propagation(X, new_weights, new_biases)

        # Calcular el error
        error = calculate_error(new_predictions, y)

        # Imprimir los errores cada vez más pequeños
        if error <= error_min:
            print(f"Iteración {i+1}: Error = {error}")
            error_min = error