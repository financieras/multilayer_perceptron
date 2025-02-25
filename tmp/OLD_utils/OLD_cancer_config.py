"""
Configuración específica para el conjunto de datos de cáncer de mama de Wisconsin.
Contiene información sobre las características, valores predeterminados y parámetros
específicos para este conjunto de datos.
"""

# Características del conjunto de datos de cáncer de mama de Wisconsin
# Las características originales son:
# - radius (promedio de distancia desde el centro a puntos del perímetro)
# - texture (desviación estándar de valores de escala de grises)
# - perimeter (perímetro del tumor)
# - area (área del tumor)
# - smoothness (variación local en longitudes de radio)
# - compactness (perímetro^2 / área - 1.0)
# - concavity (severidad de partes cóncavas del contorno)
# - concave points (número de partes cóncavas del contorno)
# - symmetry (simetría del tumor)
# - fractal dimension (dimensión fractal - aproximación al contorno a 1)

# Mapeo de nombres cortos a nombres descriptivos
FEATURE_NAMES_MAPPING = {
    'feat01': 'Radius Mean',
    'feat02': 'Texture Mean',
    'feat03': 'Perimeter Mean',
    'feat04': 'Area Mean',
    'feat05': 'Smoothness Mean',
    'feat06': 'Compactness Mean',
    'feat07': 'Concavity Mean',
    'feat08': 'Concave Points Mean',
    'feat09': 'Symmetry Mean',
    'feat10': 'Fractal Dimension Mean',
    'feat11': 'Radius SE',
    'feat12': 'Texture SE',
    'feat13': 'Perimeter SE',
    'feat14': 'Area SE',
    'feat15': 'Smoothness SE',
    'feat16': 'Compactness SE',
    'feat17': 'Concavity SE',
    'feat18': 'Concave Points SE',
    'feat19': 'Symmetry SE',
    'feat20': 'Fractal Dimension SE',
    'feat21': 'Radius Worst',
    'feat22': 'Texture Worst',
    'feat23': 'Perimeter Worst',
    'feat24': 'Area Worst',
    'feat25': 'Smoothness Worst',
    'feat26': 'Compactness Worst',
    'feat27': 'Concavity Worst',
    'feat28': 'Concave Points Worst',
    'feat29': 'Symmetry Worst',
    'feat30': 'Fractal Dimension Worst'
}

# Mapeo de diagnóstico
DIAGNOSIS_MAPPING = {'M': 0, 'B': 1}  # M: Maligno (0), B: Benigno (1)
DIAGNOSIS_LABELS = {0: 'Maligno', 1: 'Benigno'}

# Agrupación de características por categoría
FEATURE_GROUPS = {
    'Mean': ['feat01', 'feat02', 'feat03', 'feat04', 'feat05', 
            'feat06', 'feat07', 'feat08', 'feat09', 'feat10'],
    'Standard Error': ['feat11', 'feat12', 'feat13', 'feat14', 'feat15', 
                      'feat16', 'feat17', 'feat18', 'feat19', 'feat20'],
    'Worst': ['feat21', 'feat22', 'feat23', 'feat24', 'feat25', 
             'feat26', 'feat27', 'feat28', 'feat29', 'feat30']
}

# Características más importantes para el diagnóstico según la literatura
# Basado en estudios previos sobre el conjunto de datos de Wisconsin
TOP_FEATURES = [
    'feat28',  # Concave Points Worst
    'feat08',  # Concave Points Mean
    'feat27',  # Concavity Worst
    'feat23',  # Perimeter Worst
    'feat21',  # Radius Worst
    'feat03',  # Perimeter Mean
    'feat07',  # Concavity Mean
    'feat24',  # Area Worst
    'feat04',  # Area Mean
    'feat01'   # Radius Mean
]

# Colores para visualizaciones
COLORS = {
    'malignant': '#FF5555',  # Rojo para tumores malignos
    'benign': '#5555FF',     # Azul para tumores benignos
    'highlight': '#55DD55',  # Verde para destacados
    'correlation_heatmap': 'coolwarm',
    'feature_importance': 'viridis'
}

# Parámetros específicos para el modelo de cáncer de mama
CANCER_MODEL_PARAMS = {
    # Arquitectura
    'hidden_layers': 2,
    'neurons_per_layer': 16,
    'activation': 'relu',
    'output_activation': 'sigmoid',
    'weights_initializer': 'he_uniform',
    'loss': 'binary_crossentropy',
    
    # Entrenamiento
    'learning_rate': 0.005,
    'batch_size': 16,
    'epochs': 500,
    'optimizer': 'adam',
    'early_stopping': True,
    'patience': 50,
    
    # Evaluación
    'threshold': 0.5,  # Umbral para clasificación binaria
    'cv_folds': 5      # Número de folds para validación cruzada
}

# Hiperparámetros para búsqueda de grid
CANCER_GRID_SEARCH = {
    'hidden_layers': [1, 2, 3],
    'neurons_per_layer': [8, 16, 32],
    'activation': ['relu', 'tanh'],
    'learning_rate': [0.001, 0.005, 0.01],
    'batch_size': [8, 16, 32],
    'optimizer': ['adam', 'rmsprop']
}

# Nombres descriptivos para las métricas
METRIC_NAMES = {
    'accuracy': 'Precisión Global',
    'sensitivity': 'Sensibilidad (Recall)',
    'specificity': 'Especificidad',
    'precision': 'Precisión (PPV)',
    'npv': 'Valor Predictivo Negativo',
    'f1_score': 'F1-Score',
    'false_negative_rate': 'Tasa de Falsos Negativos',
    'false_positive_rate': 'Tasa de Falsos Positivos',
    'diagnostic_odds_ratio': 'Odds Ratio Diagnóstico'
}

# Información de interpretación para médicos
CLINICAL_INTERPRETATION = {
    'sensitivity': 'Porcentaje de tumores benignos correctamente identificados',
    'specificity': 'Porcentaje de tumores malignos correctamente identificados',
    'precision': 'Probabilidad de que un resultado "benigno" sea correcto',
    'npv': 'Probabilidad de que un resultado "maligno" sea correcto',
    'false_negative_rate': 'Porcentaje de tumores benignos incorrectamente identificados como malignos',
    'false_positive_rate': 'Porcentaje de tumores malignos incorrectamente identificados como benignos'
}

# Importancia clínica de las métricas (1-5, donde 5 es más importante)
METRIC_IMPORTANCE = {
    'accuracy': 3,
    'sensitivity': 5,  # Muy importante no perder casos malignos
    'specificity': 4,
    'precision': 3,
    'npv': 5,          # Muy importante confirmar correctamente malignidad
    'f1_score': 3,
    'false_negative_rate': 5,  # Crítico - casos malignos no detectados
    'false_positive_rate': 3   # Importante, pero menos crítico
}

# Umbrales de decisión recomendados según objetivo clínico
CLINICAL_THRESHOLDS = {
    'balanced': 0.5,              # Equilibrio entre sensibilidad y especificidad
    'high_sensitivity': 0.3,      # Prioriza detectar todos los malignos (menos falsos negativos)
    'high_specificity': 0.7,      # Prioriza confirmación de malignidad (menos falsos positivos)
    'optimal_f1': 0.45            # Mejor equilibrio entre precisión y recall
}