###### FUNCTIONS FOR DEEP NEURAL NETWORK ######

"""
Funciones para la implementación del Perceptrón Multicapa (MLP).
Este módulo contiene clases y funciones auxiliares para la implementación
y entrenamiento de redes neuronales artificiales.
"""

import numpy as np
import pickle
from datetime import datetime

class Activation:
    """
    Clase que contiene diferentes funciones de activación y sus derivadas.
    """
    @staticmethod
    def sigmoid(x):
        """
        Función sigmoide: f(x) = 1 / (1 + e^(-x))
        
        Parámetros:
        x (numpy.ndarray): Entrada de la función
        
        Retorna:
        numpy.ndarray: Salida de la función sigmoide
        """
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip para evitar desbordamientos
    
    @staticmethod
    def sigmoid_derivative(x):
        """
        Derivada de la función sigmoide: f'(x) = f(x) * (1 - f(x))
        
        Parámetros:
        x (numpy.ndarray): Entrada de la función
        
        Retorna:
        numpy.ndarray: Derivada de la función sigmoide
        """
        s = Activation.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def tanh(x):
        """
        Función tangente hiperbólica: f(x) = tanh(x)
        
        Parámetros:
        x (numpy.ndarray): Entrada de la función
        
        Retorna:
        numpy.ndarray: Salida de la función tangente hiperbólica
        """
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        """
        Derivada de la función tangente hiperbólica: f'(x) = 1 - tanh(x)^2
        
        Parámetros:
        x (numpy.ndarray): Entrada de la función
        
        Retorna:
        numpy.ndarray: Derivada de la función tangente hiperbólica
        """
        t = np.tanh(x)
        return 1 - t**2
    
    @staticmethod
    def relu(x):
        """
        Función ReLU: f(x) = max(0, x)
        
        Parámetros:
        x (numpy.ndarray): Entrada de la función
        
        Retorna:
        numpy.ndarray: Salida de la función ReLU
        """
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        """
        Derivada de la función ReLU: f'(x) = 1 si x > 0, 0 en otro caso
        
        Parámetros:
        x (numpy.ndarray): Entrada de la función
        
        Retorna:
        numpy.ndarray: Derivada de la función ReLU
        """
        return np.where(x > 0, 1, 0)
    
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        """
        Función Leaky ReLU: f(x) = max(alpha*x, x)
        
        Parámetros:
        x (numpy.ndarray): Entrada de la función
        alpha (float): Pendiente para valores negativos
        
        Retorna:
        numpy.ndarray: Salida de la función Leaky ReLU
        """
        return np.maximum(alpha * x, x)
    
    @staticmethod
    def leaky_relu_derivative(x, alpha=0.01):
        """
        Derivada de la función Leaky ReLU: f'(x) = 1 si x > 0, alpha en otro caso
        
        Parámetros:
        x (numpy.ndarray): Entrada de la función
        alpha (float): Pendiente para valores negativos
        
        Retorna:
        numpy.ndarray: Derivada de la función Leaky ReLU
        """
        return np.where(x > 0, 1, alpha)
    
    @staticmethod
    def softmax(x):
        """
        Función Softmax: f(x_i) = e^x_i / sum(e^x_j)
        
        Parámetros:
        x (numpy.ndarray): Entrada de la función
        
        Retorna:
        numpy.ndarray: Salida de la función Softmax
        """
        # Restar el máximo para estabilidad numérica
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    @staticmethod
    def identity(x):
        """
        Función identidad: f(x) = x
        
        Parámetros:
        x (numpy.ndarray): Entrada de la función
        
        Retorna:
        numpy.ndarray: La misma entrada sin cambios
        """
        return x
    
    @staticmethod
    def identity_derivative(x):
        """
        Derivada de la función identidad: f'(x) = 1
        
        Parámetros:
        x (numpy.ndarray): Entrada de la función
        
        Retorna:
        numpy.ndarray: Unos del mismo tamaño que la entrada
        """
        return np.ones_like(x)


class Loss:
    """
    Clase que contiene diferentes funciones de pérdida y sus derivadas.
    Incluye implementaciones para MSE, Binary Crossentropy y Categorical Crossentropy.
    """
    @staticmethod
    def mse(y_true, y_pred):
        """
        Error cuadrático medio: L = (1/n) * sum((y_true - y_pred)^2)
        
        Parámetros:
        y_true (numpy.ndarray): Valores verdaderos
        y_pred (numpy.ndarray): Valores predichos
        
        Retorna:
        float: Valor de la función de pérdida
        """
        return np.mean(np.square(y_true - y_pred))
    
    @staticmethod
    def mse_derivative(y_true, y_pred):
        """
        Derivada del error cuadrático medio: dL/dy_pred = -2 * (y_true - y_pred) / n
        
        Parámetros:
        y_true (numpy.ndarray): Valores verdaderos
        y_pred (numpy.ndarray): Valores predichos
        
        Retorna:
        numpy.ndarray: Derivada de la función de pérdida
        """
        return -2 * (y_true - y_pred) / y_true.shape[0]
    
    @staticmethod
    def binary_crossentropy(y_true, y_pred):
        """
        Entropía cruzada binaria: L = -(1/n) * sum(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
        
        Parámetros:
        y_true (numpy.ndarray): Valores verdaderos
        y_pred (numpy.ndarray): Valores predichos
        
        Retorna:
        float: Valor de la función de pérdida
        """
        # Agregar un pequeño epsilon para evitar log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def binary_crossentropy_derivative(y_true, y_pred):
        """
        Derivada de la entropía cruzada binaria: 
        dL/dy_pred = (y_pred - y_true) / (y_pred * (1 - y_pred))
        
        Parámetros:
        y_true (numpy.ndarray): Valores verdaderos
        y_pred (numpy.ndarray): Valores predichos
        
        Retorna:
        numpy.ndarray: Derivada de la función de pérdida
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.shape[0])
    
    @staticmethod
    def categorical_crossentropy(y_true, y_pred):
        """
        Entropía cruzada categórica: L = -(1/n) * sum(sum(y_true * log(y_pred)))
        
        Parámetros:
        y_true (numpy.ndarray): Valores verdaderos (codificados one-hot)
        y_pred (numpy.ndarray): Valores predichos
        
        Retorna:
        float: Valor de la función de pérdida
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    @staticmethod
    def categorical_crossentropy_derivative(y_true, y_pred):
        """
        Derivada de la entropía cruzada categórica: dL/dy_pred = -y_true / y_pred
        
        Parámetros:
        y_true (numpy.ndarray): Valores verdaderos (codificados one-hot)
        y_pred (numpy.ndarray): Valores predichos
        
        Retorna:
        numpy.ndarray: Derivada de la función de pérdida
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -y_true / (y_pred * y_true.shape[0])


class WeightInitializer:
    """
    Clase que proporciona diferentes métodos para inicializar pesos en redes neuronales.
    """
    @staticmethod
    def random(input_size, output_size):
        """
        Inicialización aleatoria de pesos entre -0.5 y 0.5
        
        Parámetros:
        input_size (int): Tamaño de la capa de entrada
        output_size (int): Tamaño de la capa de salida
        
        Retorna:
        numpy.ndarray: Matriz de pesos inicializados
        """
        return np.random.uniform(-0.5, 0.5, (input_size, output_size))
    
    @staticmethod
    def xavier_uniform(input_size, output_size):
        """
        Inicialización Xavier/Glorot uniforme:
        Rango: [-sqrt(6/(input_size + output_size)), sqrt(6/(input_size + output_size))]
        
        Parámetros:
        input_size (int): Tamaño de la capa de entrada
        output_size (int): Tamaño de la capa de salida
        
        Retorna:
        numpy.ndarray: Matriz de pesos inicializados
        """
        limit = np.sqrt(6 / (input_size + output_size))
        return np.random.uniform(-limit, limit, (input_size, output_size))
    
    @staticmethod
    def xavier_normal(input_size, output_size):
        """
        Inicialización Xavier/Glorot normal:
        Desviación: sqrt(2/(input_size + output_size))
        
        Parámetros:
        input_size (int): Tamaño de la capa de entrada
        output_size (int): Tamaño de la capa de salida
        
        Retorna:
        numpy.ndarray: Matriz de pesos inicializados
        """
        stddev = np.sqrt(2 / (input_size + output_size))
        return np.random.normal(0, stddev, (input_size, output_size))
    
    @staticmethod
    def he_uniform(input_size, output_size):
        """
        Inicialización He uniforme:
        Rango: [-sqrt(6/input_size), sqrt(6/input_size)]
        
        Parámetros:
        input_size (int): Tamaño de la capa de entrada
        output_size (int): Tamaño de la capa de salida
        
        Retorna:
        numpy.ndarray: Matriz de pesos inicializados
        """
        limit = np.sqrt(6 / input_size)
        return np.random.uniform(-limit, limit, (input_size, output_size))
    
    @staticmethod
    def he_normal(input_size, output_size):
        """
        Inicialización He normal:
        Desviación: sqrt(2/input_size)
        
        Parámetros:
        input_size (int): Tamaño de la capa de entrada
        output_size (int): Tamaño de la capa de salida
        
        Retorna:
        numpy.ndarray: Matriz de pesos inicializados
        """
        stddev = np.sqrt(2 / input_size)
        return np.random.normal(0, stddev, (input_size, output_size))
    
    @staticmethod
    def zeros(input_size, output_size):
        """
        Inicialización con ceros
        
        Parámetros:
        input_size (int): Tamaño de la capa de entrada
        output_size (int): Tamaño de la capa de salida
        
        Retorna:
        numpy.ndarray: Matriz de pesos inicializados a cero
        """
        return np.zeros((input_size, output_size))


class DenseLayer:
    """
    Implementación de una capa densa (fully connected) para redes neuronales.
    """
    def __init__(self, input_size, output_size, activation="sigmoid", weights_initializer="he_uniform"):
        """
        Inicializa una capa densa.
        
        Parámetros:
        input_size (int): Número de neuronas de entrada
        output_size (int): Número de neuronas de salida
        activation (str): Función de activación a utilizar
        weights_initializer (str): Método de inicialización de pesos
        """
        self.input_size = input_size
        self.output_size = output_size
        
        # Inicializar pesos y bias
        self.weights_initializer = weights_initializer
        self.init_weights()
        
        # Configurar función de activación
        self.activation_name = activation
        self.set_activation(activation)
        
        # Valores para el entrenamiento
        self.input = None
        self.output = None
        self.delta = None
        
        # Para optimizadores con momento o adaptabilidad
        self.velocity_w = np.zeros_like(self.weights)
        self.velocity_b = np.zeros_like(self.bias)
        self.cache_w = np.zeros_like(self.weights)
        self.cache_b = np.zeros_like(self.bias)
    
    def init_weights(self):
        """
        Inicializa los pesos y bias según el método especificado
        """
        initializers = {
            "random": WeightInitializer.random,
            "xavier_uniform": WeightInitializer.xavier_uniform,
            "xavier_normal": WeightInitializer.xavier_normal,
            "he_uniform": WeightInitializer.he_uniform,
            "he_normal": WeightInitializer.he_normal,
            "zeros": WeightInitializer.zeros
        }
        
        init_func = initializers.get(self.weights_initializer.lower(), WeightInitializer.he_uniform)
        self.weights = init_func(self.input_size, self.output_size)
        self.bias = np.zeros((1, self.output_size))
    
    def set_activation(self, activation):
        """
        Configura la función de activación y su derivada
        
        Parámetros:
        activation (str): Nombre de la función de activación
        """
        activation_funcs = {
            "sigmoid": (Activation.sigmoid, Activation.sigmoid_derivative),
            "tanh": (Activation.tanh, Activation.tanh_derivative),
            "relu": (Activation.relu, Activation.relu_derivative),
            "leaky_relu": (Activation.leaky_relu, Activation.leaky_relu_derivative),
            "softmax": (Activation.softmax, None),  # La derivada se maneja de forma especial
            "identity": (Activation.identity, Activation.identity_derivative)
        }
        
        if activation.lower() in activation_funcs:
            self.activation, self.activation_derivative = activation_funcs[activation.lower()]
        else:
            print(f"Función de activación '{activation}' no reconocida. Usando sigmoid por defecto.")
            self.activation, self.activation_derivative = activation_funcs["sigmoid"]
    
    def forward(self, input_data):
        """
        Propagación hacia adelante.
        
        Parámetros:
        input_data (numpy.ndarray): Datos de entrada
        
        Retorna:
        numpy.ndarray: Salida después de aplicar la función de activación
        """
        self.input = input_data
        self.z = np.dot(self.input, self.weights) + self.bias
        
        # Caso especial para softmax que debe aplicarse a toda la capa
        if self.activation_name.lower() == "softmax":
            self.output = Activation.softmax(self.z)
        else:
            self.output = self.activation(self.z)
        
        return self.output
    
    def backward(self, delta, learning_rate=0.01, optimizer="sgd", **kwargs):
        """
        Propagación hacia atrás.
        
        Parámetros:
        delta (numpy.ndarray): Error proveniente de la capa siguiente
        learning_rate (float): Tasa de aprendizaje
        optimizer (str): Optimizador a utilizar
        kwargs: Argumentos adicionales para el optimizador
        
        Retorna:
        numpy.ndarray: Error para la capa anterior
        """
        if self.activation_name.lower() != "softmax":
            # Para funciones de activación comunes
            delta = delta * self.activation_derivative(self.z)
        # Para softmax, el delta ya viene calculado correctamente desde la función de pérdida
        
        # Calcular el error para la capa anterior
        delta_prev = np.dot(delta, self.weights.T)
        
        # Actualizar weights y bias según el optimizador
        self.optimize(delta, learning_rate, optimizer, **kwargs)
        
        return delta_prev
    
    def optimize(self, delta, learning_rate, optimizer, **kwargs):
        """
        Actualiza los pesos y bias utilizando el optimizador especificado.
        
        Parámetros:
        delta (numpy.ndarray): Error de la capa
        learning_rate (float): Tasa de aprendizaje
        optimizer (str): Optimizador a utilizar
        kwargs: Argumentos adicionales para el optimizador
        """
        batch_size = self.input.shape[0]
        
        # Gradientes para weights y bias
        dw = np.dot(self.input.T, delta) / batch_size
        db = np.mean(delta, axis=0, keepdims=True)
        
        if optimizer.lower() == "sgd":
            # Descenso de gradiente estocástico (SGD)
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db
        
        elif optimizer.lower() == "momentum":
            # SGD con momento
            momentum = kwargs.get("momentum", 0.9)
            self.velocity_w = momentum * self.velocity_w - learning_rate * dw
            self.velocity_b = momentum * self.velocity_b - learning_rate * db
            self.weights += self.velocity_w
            self.bias += self.velocity_b
        
        elif optimizer.lower() == "nesterov":
            # Momento de Nesterov
            momentum = kwargs.get("momentum", 0.9)
            v_prev_w = self.velocity_w.copy()
            v_prev_b = self.velocity_b.copy()
            
            self.velocity_w = momentum * self.velocity_w - learning_rate * dw
            self.velocity_b = momentum * self.velocity_b - learning_rate * db
            
            self.weights += -momentum * v_prev_w + (1 + momentum) * self.velocity_w
            self.bias += -momentum * v_prev_b + (1 + momentum) * self.velocity_b
        
        elif optimizer.lower() == "rmsprop":
            # RMSprop
            decay = kwargs.get("decay", 0.9)
            epsilon = kwargs.get("epsilon", 1e-8)
            
            self.cache_w = decay * self.cache_w + (1 - decay) * np.square(dw)
            self.cache_b = decay * self.cache_b + (1 - decay) * np.square(db)
            
            self.weights -= learning_rate * dw / (np.sqrt(self.cache_w) + epsilon)
            self.bias -= learning_rate * db / (np.sqrt(self.cache_b) + epsilon)
        
        elif optimizer.lower() == "adam":
            # Adam
            beta1 = kwargs.get("beta1", 0.9)
            beta2 = kwargs.get("beta2", 0.999)
            epsilon = kwargs.get("epsilon", 1e-8)
            t = kwargs.get("t", 1)  # Iteración actual
            
            # Actualizar momentos
            self.velocity_w = beta1 * self.velocity_w + (1 - beta1) * dw
            self.velocity_b = beta1 * self.velocity_b + (1 - beta1) * db
            
            # Actualizar cache
            self.cache_w = beta2 * self.cache_w + (1 - beta2) * np.square(dw)
            self.cache_b = beta2 * self.cache_b + (1 - beta2) * np.square(db)
            
            # Corregir el bias
            v_corrected_w = self.velocity_w / (1 - beta1**t)
            v_corrected_b = self.velocity_b / (1 - beta1**t)
            c_corrected_w = self.cache_w / (1 - beta2**t)
            c_corrected_b = self.cache_b / (1 - beta2**t)
            
            # Actualizar pesos
            self.weights -= learning_rate * v_corrected_w / (np.sqrt(c_corrected_w) + epsilon)
            self.bias -= learning_rate * v_corrected_b / (np.sqrt(c_corrected_b) + epsilon)
        
        else:
            # Si no se reconoce el optimizador, usar SGD por defecto
            print(f"Optimizador '{optimizer}' no reconocido. Usando SGD por defecto.")
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db


class MultilayerPerceptron:
    """
    Implementación de un Perceptrón Multicapa (MLP).
    """
    def __init__(self):
        """Inicializa un Perceptrón Multicapa vacío"""
        self.layers = []
        self.loss_func = None
        self.loss_derivative = None
        self.training_loss_history = []
        self.validation_loss_history = []
        self.training_accuracy_history = []
        self.validation_accuracy_history = []
    
    def add_layer(self, layer):
        """
        Agrega una capa a la red.
        
        Parámetros:
        layer (DenseLayer): Capa a agregar
        """
        self.layers.append(layer)
    
    def set_loss(self, loss_name):
        """
        Configura la función de pérdida.
        
        Parámetros:
        loss_name (str): Nombre de la función de pérdida
        """
        loss_funcs = {
            "mse": (Loss.mse, Loss.mse_derivative),
            "binary_crossentropy": (Loss.binary_crossentropy, Loss.binary_crossentropy_derivative),
            "categorical_crossentropy": (Loss.categorical_crossentropy, Loss.categorical_crossentropy_derivative)
        }
        
        if loss_name.lower() in loss_funcs:
            self.loss_func, self.loss_derivative = loss_funcs[loss_name.lower()]
        else:
            print(f"Función de pérdida '{loss_name}' no reconocida. Usando MSE por defecto.")
            self.loss_func, self.loss_derivative = loss_funcs["mse"]
    
    def forward(self, X):
        """
        Propagación hacia adelante en todas las capas.
        
        Parámetros:
        X (numpy.ndarray): Datos de entrada
        
        Retorna:
        numpy.ndarray: Salida de la red
        """
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, y_true, y_pred, learning_rate, optimizer="sgd", **kwargs):
        """
        Propagación hacia atrás en todas las capas.
        
        Parámetros:
        y_true (numpy.ndarray): Valores verdaderos
        y_pred (numpy.ndarray): Valores predichos
        learning_rate (float): Tasa de aprendizaje
        optimizer (str): Optimizador a utilizar
        kwargs: Argumentos adicionales para el optimizador
        """
        # Calcular el error inicial
        delta = self.loss_derivative(y_true, y_pred)
        
        # Propagar el error hacia atrás en cada capa
        for layer in reversed(self.layers):
            delta = layer.backward(delta, learning_rate, optimizer, **kwargs)
    
    def train_batch(self, X_batch, y_batch, learning_rate, optimizer="sgd", **kwargs):
        """
        Entrena la red en un lote de datos.
        
        Parámetros:
        X_batch (numpy.ndarray): Características del lote
        y_batch (numpy.ndarray): Etiquetas del lote
        learning_rate (float): Tasa de aprendizaje
        optimizer (str): Optimizador a utilizar
        kwargs: Argumentos adicionales para el optimizador
        
        Retorna:
        float: Pérdida en este lote
        """
        # Propagación hacia adelante
        y_pred = self.forward(X_batch)
        
        # Calcular pérdida
        loss = self.loss_func(y_batch, y_pred)
        
        # Propagación hacia atrás
        self.backward(y_batch, y_pred, learning_rate, optimizer, **kwargs)
        
        return loss
    
    def predict(self, X):
        """
        Realiza predicciones con la red entrenada.
        
        Parámetros:
        X (numpy.ndarray): Datos de entrada
        
        Retorna:
        numpy.ndarray: Predicciones
        """
        return self.forward(X)
    
    def predict_classes(self, X, threshold=0.5):
        """
        Predice las clases para datos binarios.
        
        Parámetros:
        X (numpy.ndarray): Datos de entrada
        threshold (float): Umbral para clasificación binaria
        
        Retorna:
        numpy.ndarray: Clases predichas (0 o 1)
        """
        predictions = self.predict(X)
        return (predictions > threshold).astype(int)
    
    def evaluate(self, X, y_true, threshold=0.5):
        """
        Evalúa el rendimiento de la red.
        
        Parámetros:
        X (numpy.ndarray): Datos de entrada
        y_true (numpy.ndarray): Valores verdaderos
        threshold (float): Umbral para clasificación binaria
        
        Retorna:
        dict: Métricas de rendimiento (pérdida, precisión)
        """
        y_pred = self.predict(X)
        loss = self.loss_func(y_true, y_pred)
        
        # Para clasificación binaria
        y_pred_class = (y_pred > threshold).astype(int)
        accuracy = np.mean(y_pred_class == y_true)
        
        return {
            "loss": loss,
            "accuracy": accuracy
        }
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, 
            epochs=100, batch_size=32, learning_rate=0.01, 
            optimizer="sgd", verbose=1, early_stopping=False, 
            patience=10, min_delta=0.001, **kwargs):
        """
        Entrena la red neuronal.
        
        Parámetros:
        X_train (numpy.ndarray): Características de entrenamiento
        y_train (numpy.ndarray): Etiquetas de entrenamiento
        X_val (numpy.ndarray): Características de validación
        y_val (numpy.ndarray): Etiquetas de validación
        epochs (int): Número de épocas
        batch_size (int): Tamaño del lote
        learning_rate (float): Tasa de aprendizaje
        optimizer (str): Optimizador a utilizar
        verbose (int): Nivel de detalle de la salida
        early_stopping (bool): Si se usa parada temprana
        patience (int): Número de épocas para parada temprana
        min_delta (float): Cambio mínimo para considerar mejora
        kwargs: Argumentos adicionales para el optimizador
        
        Retorna:
        dict: Historial de entrenamiento
        """
        n_samples = X_train.shape[0]
        
        # Reiniciar historiales
        self.training_loss_history = []
        self.validation_loss_history = []
        self.training_accuracy_history = []
        self.validation_accuracy_history = []
        
        # Para parada temprana
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Iterar sobre las épocas
        for epoch in range(epochs):
            # Mezclar los datos de entrenamiento
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Entrenar por lotes
            epoch_loss = 0
            n_batches = int(np.ceil(n_samples / batch_size))
            
            for batch in range(n_batches):
                start = batch * batch_size
                end = min(start + batch_size, n_samples)
                
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                # Convertir y_batch a formato adecuado (columna)
                if len(y_batch.shape) == 1:
                    y_batch = y_batch.reshape(-1, 1)
                
                # Entrenar este lote
                t = epoch * n_batches + batch + 1  # Iteración actual para Adam
                batch_loss = self.train_batch(X_batch, y_batch, learning_rate, optimizer, t=t, **kwargs)
                epoch_loss += batch_loss
            
            # Calcular pérdida y precisión promedio de la época
            epoch_loss /= n_batches
            self.training_loss_history.append(epoch_loss)
            
            # Evaluar en conjunto de entrenamiento
            train_metrics = self.evaluate(X_train, y_train.reshape(-1, 1))
            self.training_accuracy_history.append(train_metrics["accuracy"])
            
            # Evaluar en conjunto de validación si se proporciona
            val_metrics = {"loss": None, "accuracy": None}
            if X_val is not None and y_val is not None:
                val_metrics = self.evaluate(X_val, y_val.reshape(-1, 1))
                self.validation_loss_history.append(val_metrics["loss"])
                self.validation_accuracy_history.append(val_metrics["accuracy"])
                
                # Parada temprana
                if early_stopping:
                    if val_metrics["loss"] < best_val_loss - min_delta:
                        best_val_loss = val_metrics["loss"]
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            if verbose > 0:
                                print(f"\nParada temprana en época {epoch+1}")
                            break
            
            # Imprimir progreso
            if verbose > 0 and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                val_loss_str = f"- val_loss: {val_metrics['loss']:.4f}" if val_metrics["loss"] is not None else ""
                val_acc_str = f"- val_acc: {val_metrics['accuracy']:.4f}" if val_metrics["accuracy"] is not None else ""
                print(f"Época {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - acc: {train_metrics['accuracy']:.4f} {val_loss_str} {val_acc_str}")
        
        if verbose > 0:
            print("\nEntrenamiento completado.")
        
        return {
            "training_loss": self.training_loss_history,
            "validation_loss": self.validation_loss_history,
            "training_accuracy": self.training_accuracy_history,
            "validation_accuracy": self.validation_accuracy_history
        }
    
    def save(self, filepath):
        """
        Guarda el modelo en un archivo.
        
        Parámetros:
        filepath (str): Ruta donde guardar el modelo
        """
        # Crear un diccionario con la estructura y pesos del modelo
        model_data = {
            "architecture": [
                {
                    "type": "DenseLayer",
                    "input_size": layer.input_size,
                    "output_size": layer.output_size,
                    "activation": layer.activation_name,
                    "weights_initializer": layer.weights_initializer,
                    "weights": layer.weights.tolist(),
                    "bias": layer.bias.tolist()
                } for layer in self.layers
            ],
            "loss": self.loss_func.__name__ if hasattr(self.loss_func, "__name__") else None,
            "history": {
                "training_loss": self.training_loss_history,
                "validation_loss": self.validation_loss_history,
                "training_accuracy": self.training_accuracy_history,
                "validation_accuracy": self.validation_accuracy_history
            },
            "metadata": {
                "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        # Guardar como archivo pickle
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
        
        print(f"Modelo guardado en: {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Carga un modelo desde un archivo.
        
        Parámetros:
        filepath (str): Ruta del archivo del modelo
        
        Retorna:
        MultilayerPerceptron: Modelo cargado
        """
        # Cargar el diccionario del modelo
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)
        
        # Crear un nuevo modelo
        model = cls()
        
        # Configurar la función de pérdida
        if model_data.get("loss"):
            model.set_loss(model_data["loss"])
        
        # Recrear las capas
        for layer_data in model_data["architecture"]:
            if layer_data["type"] == "DenseLayer":
                layer = DenseLayer(
                    input_size=layer_data["input_size"],
                    output_size=layer_data["output_size"],
                    activation=layer_data["activation"],
                    weights_initializer=layer_data["weights_initializer"]
                )
                
                # Cargar pesos y bias
                layer.weights = np.array(layer_data["weights"])
                layer.bias = np.array(layer_data["bias"])
                
                # Agregar la capa al modelo
                model.add_layer(layer)
        
        # Cargar historial si existe
        if "history" in model_data:
            model.training_loss_history = model_data["history"].get("training_loss", [])
            model.validation_loss_history = model_data["history"].get("validation_loss", [])
            model.training_accuracy_history = model_data["history"].get("training_accuracy", [])
            model.validation_accuracy_history = model_data["history"].get("validation_accuracy", [])
        
        return model


# Funciones auxiliares para crear y configurar redes neuronales

def create_mlp(input_size, output_size, hidden_layers=2, neurons_per_layer=25,
               activation="sigmoid", output_activation="sigmoid", 
               weights_initializer="he_uniform", loss="binary_crossentropy"):
    """
    Crea un Perceptrón Multicapa con la configuración especificada.
    
    Parámetros:
    input_size (int): Tamaño de la capa de entrada
    output_size (int): Tamaño de la capa de salida
    hidden_layers (int): Número de capas ocultas
    neurons_per_layer (int): Número de neuronas por capa oculta
    activation (str): Función de activación para las capas ocultas
    output_activation (str): Función de activación para la capa de salida
    weights_initializer (str): Método de inicialización de pesos
    loss (str): Función de pérdida
    
    Retorna:
    MultilayerPerceptron: Modelo configurado
    """
    model = MultilayerPerceptron()
    
    # Capa de entrada a primera capa oculta
    model.add_layer(DenseLayer(
        input_size=input_size,
        output_size=neurons_per_layer,
        activation=activation,
        weights_initializer=weights_initializer
    ))
    
    # Capas ocultas adicionales
    for _ in range(hidden_layers - 1):
        model.add_layer(DenseLayer(
            input_size=neurons_per_layer,
            output_size=neurons_per_layer,
            activation=activation,
            weights_initializer=weights_initializer
        ))
    
    # Capa de salida
    model.add_layer(DenseLayer(
        input_size=neurons_per_layer,
        output_size=output_size,
        activation=output_activation,
        weights_initializer=weights_initializer
    ))
    
    # Configurar la función de pérdida
    model.set_loss(loss)
    
    return model


def create_custom_mlp(layer_sizes, activations, weights_initializers, loss="binary_crossentropy"):
    """
    Crea un Perceptrón Multicapa personalizado con configuración específica para cada capa.
    
    Parámetros:
    layer_sizes (list): Lista con el tamaño de cada capa, incluyendo entrada y salida
    activations (list): Lista con la función de activación para cada capa
    weights_initializers (list): Lista con el método de inicialización de pesos para cada capa
    loss (str): Función de pérdida
    
    Retorna:
    MultilayerPerceptron: Modelo configurado
    """
    if len(layer_sizes) < 2:
        raise ValueError("Se necesitan al menos 2 capas (entrada y salida)")
    
    if len(activations) != len(layer_sizes) - 1:
        raise ValueError("El número de funciones de activación debe ser igual al número de capas - 1")
    
    if len(weights_initializers) != len(layer_sizes) - 1:
        raise ValueError("El número de inicializadores de pesos debe ser igual al número de capas - 1")
    
    model = MultilayerPerceptron()
    
    # Crear cada capa
    for i in range(len(layer_sizes) - 1):
        model.add_layer(DenseLayer(
            input_size=layer_sizes[i],
            output_size=layer_sizes[i + 1],
            activation=activations[i],
            weights_initializer=weights_initializers[i]
        ))
    
    # Configurar la función de pérdida
    model.set_loss(loss)
    
    return model


def experiment_with_hyperparameters(X_train, y_train, X_test, y_test,
                                   hidden_layers_options=[1, 2, 3],
                                   neurons_options=[15, 25, 35],
                                   activation_options=["sigmoid", "tanh", "relu"],
                                   optimizer_options=["sgd", "momentum", "adam"],
                                   learning_rate_options=[0.001, 0.01, 0.1],
                                   epochs=100,
                                   batch_size=32,
                                   early_stopping=True,
                                   patience=5):
    """
    Experimenta con diferentes hiperparámetros y encuentra la mejor configuración.
    
    Parámetros:
    X_train, y_train: Datos de entrenamiento
    X_test, y_test: Datos de prueba
    hidden_layers_options: Lista de opciones para el número de capas ocultas
    neurons_options: Lista de opciones para el número de neuronas por capa
    activation_options: Lista de opciones para las funciones de activación
    optimizer_options: Lista de opciones para los optimizadores
    learning_rate_options: Lista de opciones para la tasa de aprendizaje
    epochs: Número máximo de épocas
    batch_size: Tamaño del lote
    early_stopping: Si se usa parada temprana
    patience: Paciencia para parada temprana
    
    Retorna:
    dict: Mejor configuración y su rendimiento
    """
    best_accuracy = 0
    best_config = {}
    results = []
    
    # Total de experimentos
    total_experiments = (len(hidden_layers_options) * 
                        len(neurons_options) * 
                        len(activation_options) * 
                        len(optimizer_options) * 
                        len(learning_rate_options))
    
    print(f"Iniciando {total_experiments} experimentos...")
    
    experiment_count = 0
    
    # Iterar sobre todas las combinaciones
    for hidden_layers in hidden_layers_options:
        for neurons in neurons_options:
            for activation in activation_options:
                for optimizer in optimizer_options:
                    for lr in learning_rate_options:
                        experiment_count += 1
                        print(f"\nExperimento {experiment_count}/{total_experiments}")
                        print(f"Configuración: {hidden_layers} capas, {neurons} neuronas, {activation}, {optimizer}, lr={lr}")
                        
                        # Crear y entrenar modelo
                        model = create_mlp(
                            input_size=X_train.shape[1],
                            output_size=1,
                            hidden_layers=hidden_layers,
                            neurons_per_layer=neurons,
                            activation=activation,
                            output_activation="sigmoid",
                            weights_initializer="he_uniform",
                            loss="binary_crossentropy"
                        )
                        
                        # Entrenar con early stopping para ahorrar tiempo
                        history = model.fit(
                            X_train=X_train,
                            y_train=y_train,
                            X_val=X_test,
                            y_val=y_test,
                            epochs=epochs,
                            batch_size=batch_size,
                            learning_rate=lr,
                            optimizer=optimizer,
                            early_stopping=early_stopping,
                            patience=patience,
                            verbose=0
                        )
                        
                        # Evaluar en conjunto de prueba
                        test_metrics = model.evaluate(X_test, y_test.reshape(-1, 1))
                        test_accuracy = test_metrics["accuracy"]
                        test_loss = test_metrics["loss"]
                        
                        print(f"Resultado: Precisión={test_accuracy:.4f}, Pérdida={test_loss:.4f}")
                        
                        # Guardar resultados
                        config = {
                            "hidden_layers": hidden_layers,
                            "neurons": neurons,
                            "activation": activation,
                            "optimizer": optimizer,
                            "learning_rate": lr,
                            "accuracy": test_accuracy,
                            "loss": test_loss,
                            "epochs_trained": len(history["training_loss"])
                        }
                        
                        results.append(config)
                        
                        # Actualizar mejor configuración
                        if test_accuracy > best_accuracy:
                            best_accuracy = test_accuracy
                            best_config = config
                            print(f"¡Nueva mejor configuración! Precisión: {best_accuracy:.4f}")
    
    return {
        "best_config": best_config,
        "all_results": results
    }