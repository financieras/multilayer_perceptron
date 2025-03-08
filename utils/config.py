# Configuración y parámetros del proyecto

# Diccionario con los nombres completos de las características del Wisconsin Breast Cancer Dataset
FEATURE_NAMES = {
    'f01': 'mean radius',
    'f02': 'mean texture',
    'f03': 'mean perimeter',
    'f04': 'mean area',
    'f05': 'mean smoothness',
    'f06': 'mean compactness',
    'f07': 'mean concavity',
    'f08': 'mean concave points',
    'f09': 'mean symmetry',
    'f10': 'mean fractal dimension',
    'f11': 'radius error',
    'f12': 'texture error',
    'f13': 'perimeter error',
    'f14': 'area error',
    'f15': 'smoothness error',
    'f16': 'compactness error',
    'f17': 'concavity error',
    'f18': 'concave points error',
    'f19': 'symmetry error',
    'f20': 'fractal dimension error',
    'f21': 'worst radius',
    'f22': 'worst texture',
    'f23': 'worst perimeter',
    'f24': 'worst area',
    'f25': 'worst smoothness',
    'f26': 'worst compactness',
    'f27': 'worst concavity',
    'f28': 'worst concave points',
    'f29': 'worst symmetry',
    'f30': 'worst fractal dimension'
}


# Arquitectura de la red neuronal
HIDDEN_LAYERS = [30, 15, 8]  # Se pueden poner cualquier número de capas/neuronas


RANDOM_STATE = 42      # Semilla para reproducibilidad
LEARNING_RATE = 0.01   # Se pueden usar otros valores como 0.01, 0.001, etc.
EPOCHS = 1000          # Establecer el número de epócas del entrenamiento del modelo
BATCH_SIZE = 8         # Definimos el BATCH_SIZE
STRATIFY = True        # Controla si la división de los datos mantiene la proporción de clases

# Tasa de decaimiento del learning rate
LR_DECAY = 0.95  # Reduce la tasa de aprendizaje en un 5% cada cierto número de épocas

