import numpy as np
import json
import os

# Funciones de activación
def sigmoid(x):
    """Función de activación sigmoide"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip para evitar overflow

def sigmoid_derivative(x):
    """Derivada de la función sigmoide"""
    s = sigmoid(x)
    return s * (1 - s)

def softmax(x):
    """Función de activación softmax para la capa de salida"""
    # Restar el máximo para estabilidad numérica
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def binary_crossentropy(y_true, y_pred):
    """Función de pérdida binary crossentropy"""
    # Añadir epsilon para evitar log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Inicializadores de pesos
def he_uniform(shape):
    """Inicialización He Uniform para los pesos"""
    fan_in = shape[0]
    limit = np.sqrt(6 / fan_in)
    return np.random.uniform(-limit, limit, shape)

def xavier_uniform(shape):
    """Inicialización Xavier Uniform para los pesos"""
    fan_in, fan_out = shape
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, shape)

# Clase para la capa densa
class DenseLayer:
    def __init__(self, input_dim, output_dim, activation="sigmoid", weights_initializer="he_uniform"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Inicializar pesos y sesgos
        if weights_initializer == "he_uniform":
            self.weights = he_uniform((input_dim, output_dim))
        elif weights_initializer == "xavier_uniform":
            self.weights = xavier_uniform((input_dim, output_dim))
        else:
            # Inicialización simple por defecto
            self.weights = np.random.randn(input_dim, output_dim) * 0.01
            
        self.bias = np.zeros((1, output_dim))
        
        # Definir la función de activación
        if activation == "sigmoid":
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == "softmax":
            self.activation = softmax
            # No necesitamos la derivada de softmax para la capa de salida con crossentropy
            self.activation_derivative = None
        else:
            raise ValueError(f"Activación {activation} no implementada")
            
        self.activation_name = activation
        
        # Variables para almacenar valores durante el forward y backward pass
        self.input = None
        self.z = None  # Pre-activación
        self.output = None  # Post-activación
        
    def forward(self, input_data):
        """Forward pass: calcular la salida de la capa"""
        self.input = input_data
        self.z = np.dot(input_data, self.weights) + self.bias
        self.output = self.activation(self.z)
        return self.output
    
    def backward(self, dL_dout, learning_rate):
        """Backward pass: actualizar pesos y sesgos, devolver gradiente para la capa anterior"""
        batch_size = self.input.shape[0]
        
        if self.activation_name == "softmax":
            # Para softmax con binary crossentropy, el gradiente se simplifica
            dL_dz = dL_dout
        else:
            # Para otras activaciones, calcular dL/dz = dL/dout * d(out)/dz
            dL_dz = dL_dout * self.activation_derivative(self.z)
        
        # Calcular gradientes
        dL_dw = np.dot(self.input.T, dL_dz) / batch_size
        dL_db = np.mean(dL_dz, axis=0, keepdims=True)
        dL_dinput = np.dot(dL_dz, self.weights.T)
        
        # Actualizar pesos y sesgos
        self.weights -= learning_rate * dL_dw
        self.bias -= learning_rate * dL_db
        
        return dL_dinput

# Clase para el modelo de red neuronal
class NeuralNetwork:
    def __init__(self):
        self.layers = []
    
    def add(self, layer):
        """Añadir una capa a la red"""
        self.layers.append(layer)
    
    def forward(self, X):
        """Forward pass a través de todas las capas"""
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, dL_dout, learning_rate):
        """Backward pass a través de todas las capas"""
        gradient = dL_dout
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)
    

    def train(self, X_train, y_train, X_valid=None, y_valid=None, epochs=100, batch_size=32, learning_rate=0.01, lr_decay=0.95):
        """Entrenar la red neuronal"""
        num_samples = X_train.shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))
        
        train_losses = []
        valid_losses = []
        train_accuracies = []
        valid_accuracies = []
        current_lr = learning_rate
        
        for epoch in range(epochs):
            # Aplicar decay a la tasa de aprendizaje
            if epoch > 0 and epoch % 10 == 0:
                current_lr *= lr_decay
                print(f"Tasa de aprendizaje ajustada a: {current_lr:.6f}")
            
            # Mezclar los datos de entrenamiento
            indices = np.random.permutation(num_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            
            # Entrenamiento por mini-batches
            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, num_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                
                # Calcular pérdida
                batch_loss = binary_crossentropy(y_batch, y_pred)
                epoch_loss += batch_loss * (end_idx - start_idx) / num_samples
                
                # Calcular gradiente de la función de pérdida respecto a la salida
                # Para binary crossentropy con sigmoid, el gradiente es (y_pred - y_true)
                dL_dout = y_pred - y_batch
                
                # Backward pass
                self.backward(dL_dout, current_lr)
            
            train_losses.append(epoch_loss)
            
            # Calcular accuracy en el conjunto de entrenamiento
            train_preds = self.predict(X_train)
            train_acc = calculate_accuracy(y_train, train_preds)
            train_accuracies.append(train_acc)
            
            # Calcular métricas en el conjunto de validación si está disponible
            valid_loss = None
            valid_acc = None
            if X_valid is not None and y_valid is not None:
                y_valid_pred = self.forward(X_valid)
                valid_loss = binary_crossentropy(y_valid, y_valid_pred)
                valid_losses.append(valid_loss)
                
                valid_acc = calculate_accuracy(y_valid, y_valid_pred)
                valid_accuracies.append(valid_acc)
            
            # Imprimir progreso
            if valid_loss is not None:
                print(f"epoch {epoch+1:02d}/{epochs} - loss: {epoch_loss:.4f} - acc: {train_acc:.4f} - val_loss: {valid_loss:.4f} - val_acc: {valid_acc:.4f}")
            else:
                print(f"epoch {epoch+1:02d}/{epochs} - loss: {epoch_loss:.4f} - acc: {train_acc:.4f}")
        
        return train_losses, valid_losses, train_accuracies, valid_accuracies

    
    def predict(self, X):
        """Realizar predicciones"""
        return self.forward(X)
    
    def save(self, filepath):
        """Guardar los pesos del modelo"""
        model_params = {
            'architecture': [],
            'weights': [],
            'biases': []
        }
        
        for i, layer in enumerate(self.layers):
            model_params['architecture'].append({
                'layer_type': 'dense',
                'input_dim': layer.input_dim,
                'output_dim': layer.output_dim,
                'activation': layer.activation_name
            })
            model_params['weights'].append(layer.weights.tolist())
            model_params['biases'].append(layer.bias.tolist())
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Guardar modelo como JSON
        with open(filepath, 'w') as f:
            json.dump(model_params, f)
        
        print(f"Modelo guardado en {filepath}")
    
    def load(self, filepath):
        """Cargar los pesos del modelo"""
        with open(filepath, 'r') as f:
            model_params = json.load(f)
        
        self.layers = []
        
        for i, layer_params in enumerate(model_params['architecture']):
            layer = DenseLayer(
                input_dim=layer_params['input_dim'],
                output_dim=layer_params['output_dim'],
                activation=layer_params['activation']
            )
            layer.weights = np.array(model_params['weights'][i])
            layer.bias = np.array(model_params['biases'][i])
            self.layers.append(layer)
        
        print(f"Modelo cargado desde {filepath}")

    # Método para evaluar el modelo
    def evaluate(self, X, y):
        """
        Evalúa el modelo en un conjunto de datos
        
        Args:
            X: Datos de entrada
            y: Etiquetas verdaderas
            
        Returns:
            tuple: (loss, accuracy)
        """
        y_pred = self.predict(X)
        loss = binary_crossentropy(y, y_pred)
        accuracy = calculate_accuracy(y, y_pred)
        return loss, accuracy

        

# Función para crear una red neuronal con la arquitectura especificada
def create_network(input_shape, hidden_layers, output_shape):
    """Crear una red neuronal con la arquitectura especificada"""
    model = NeuralNetwork()
    
    # Añadir primera capa oculta
    model.add(DenseLayer(input_shape, hidden_layers[0], activation="sigmoid", weights_initializer="xavier_uniform"))
    
    # Añadir capas ocultas adicionales
    for i in range(1, len(hidden_layers)):
        model.add(DenseLayer(hidden_layers[i-1], hidden_layers[i], activation="sigmoid", weights_initializer="xavier_uniform"))
    
    # Capa de salida
    model.add(DenseLayer(hidden_layers[-1], output_shape, activation="sigmoid", weights_initializer="xavier_uniform"))
    
    return model



# Cálculo de Accuracy
def calculate_accuracy(y_true, y_pred):
    """
    Calcula la exactitud (accuracy) para clasificación binaria
    
    Args:
        y_true: Etiquetas verdaderas (0 o 1)
        y_pred: Probabilidades predichas
        
    Returns:
        float: Accuracy (0.0 a 1.0)
    """
    # Convertir probabilidades a predicciones binarias (0 o 1)
    predictions = (y_pred > 0.5).astype(int)
    # Calcular accuracy
    return np.mean(predictions == y_true)


