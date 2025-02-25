"""
Archivo de configuración para el proyecto de Perceptrón Multicapa.
Contiene constantes y parámetros utilizados en diferentes módulos del proyecto.
"""

import os

# Rutas de directorios
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

# Asegurar que los directorios existan
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR, FIGURES_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Parámetros de normalización
NORMALIZATION_METHODS = ["zscore", "minmax", "robust", "log"]
DEFAULT_NORMALIZATION = "zscore"

# Parámetros de división de datos
TRAIN_TEST_SPLIT = {
    "train_size": 0.8,
    "test_size": 0.2,
    "random_state": 42,
    "stratify": True
}

# Parámetros por defecto para el Perceptrón Multicapa
DEFAULT_MLP_PARAMS = {
    # Arquitectura
    "hidden_layers": 2,
    "neurons_per_layer": 25,
    "activation": "sigmoid",
    "output_activation": "sigmoid",
    "weights_initializer": "he_uniform",
    "loss": "binary_crossentropy",
    
    # Entrenamiento
    "learning_rate": 0.01,
    "batch_size": 32,
    "epochs": 100,
    "optimizer": "adam",
    "early_stopping": True,
    "patience": 10,
    "min_delta": 0.001,
    
    # Parámetros específicos de optimizadores
    "momentum": 0.9,  # Para momentum y nesterov
    "beta1": 0.9,     # Para adam
    "beta2": 0.999,   # Para adam
    "epsilon": 1e-8,  # Para rmsprop y adam
    "decay": 0.9      # Para rmsprop
}

# Opciones de hiperparámetros para experimentación
HYPERPARAMETER_OPTIONS = {
    "hidden_layers": [1, 2, 3, 4],
    "neurons_per_layer": [10, 15, 25, 35, 50],
    "activation": ["sigmoid", "tanh", "relu", "leaky_relu"],
    "optimizer": ["sgd", "momentum", "nesterov", "rmsprop", "adam"],
    "learning_rate": [0.001, 0.005, 0.01, 0.05, 0.1],
    "batch_size": [8, 16, 32, 64, 128]
}

# Parámetros de evaluación del modelo
EVALUATION_METRICS = ["accuracy", "precision", "recall", "f1", "auc", "confusion_matrix"]

# Nombres de características del dataset
FEATURE_NAMES = [f'feat{str(i+1).zfill(2)}' for i in range(30)]

# Mapeo de diagnóstico
DIAGNOSIS_MAPPING = {'M': 0, 'B': 1}
DIAGNOSIS_LABELS = {0: 'Maligno', 1: 'Benigno'}

# Colores para visualizaciones
COLORS = {
    'malignant': '#FF5555',
    'benign': '#5555FF',
    'correlation_heatmap': 'coolwarm',
    'train': '#1f77b4',
    'test': '#ff7f0e',
    'validation': '#2ca02c'
}

# Configuración de gráficos
FIGURE_DPI = 300
FIGURE_SIZE = {
    'small': (8, 6),
    'medium': (12, 8),
    'large': (16, 10),
    'wide': (18, 6)
}