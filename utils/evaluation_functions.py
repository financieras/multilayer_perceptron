import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def confusion_matrix(y_true, y_pred):
    """
    Calcula la matriz de confusión para clasificación binaria.
    
    Args:
        y_true: Etiquetas verdaderas (0 o 1)
        y_pred: Etiquetas predichas (0 o 1)
        
    Returns:
        numpy.ndarray: Matriz de confusión con formato [[TN, FP], [FN, TP]]
    """
    # Asegurarse de que y_true y y_pred son arrays unidimensionales
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Contar verdaderos negativos (TN), falsos positivos (FP), falsos negativos (FN) y verdaderos positivos (TP)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    
    # Construir la matriz de confusión
    return np.array([[tn, fp], [fn, tp]])

def precision_score(y_true, y_pred):
    """
    Calcula la precisión (precision) para clasificación binaria.
    precision = TP / (TP + FP)
    
    Args:
        y_true: Etiquetas verdaderas (0 o 1)
        y_pred: Etiquetas predichas (0 o 1)
        
    Returns:
        float: Precisión (de 0.0 a 1.0)
    """
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]
    fp = cm[0, 1]
    
    # Evitar división por cero
    if tp + fp == 0:
        return 0.0
    
    return tp / (tp + fp)

def recall_score(y_true, y_pred):
    """
    Calcula la exhaustividad (recall) para clasificación binaria.
    recall = TP / (TP + FN)
    
    Args:
        y_true: Etiquetas verdaderas (0 o 1)
        y_pred: Etiquetas predichas (0 o 1)
        
    Returns:
        float: Recall (de 0.0 a 1.0)
    """
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]
    fn = cm[1, 0]
    
    # Evitar división por cero
    if tp + fn == 0:
        return 0.0
    
    return tp / (tp + fn)

def f1_score(y_true, y_pred):
    """
    Calcula el F1-score para clasificación binaria.
    F1 = 2 * (precision * recall) / (precision + recall)
    
    Args:
        y_true: Etiquetas verdaderas (0 o 1)
        y_pred: Etiquetas predichas (0 o 1)
        
    Returns:
        float: F1-score (de 0.0 a 1.0)
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    # Evitar división por cero
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)

def plot_confusion_matrix(y_true, y_pred, figsize=(8, 6), save_path=None):
    """
    Visualiza la matriz de confusión para clasificación binaria.
    
    Args:
        y_true: Etiquetas verdaderas (0 o 1)
        y_pred: Etiquetas predichas (0 o 1)
        figsize: Tamaño de la figura (ancho, alto)
        save_path: Ruta para guardar la imagen (opcional)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benigno (0)', 'Maligno (1)'],
                yticklabels=['Benigno (0)', 'Maligno (1)'])
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_prediction_distribution(probabilities, y_true, figsize=(10, 6), save_path=None):
    """
    Visualiza la distribución de probabilidades predichas.
    
    Args:
        probabilities: Probabilidades predichas (valores continuos entre 0 y 1)
        y_true: Etiquetas verdaderas (0 o 1)
        figsize: Tamaño de la figura (ancho, alto)
        save_path: Ruta para guardar la imagen (opcional)
    """
    plt.figure(figsize=figsize)
    
    # Aplanar los arrays para asegurarse de que son unidimensionales
    probabilities = probabilities.flatten()
    y_true = y_true.flatten()
    
    # Crear DataFrame temporal para seaborn
    import pandas as pd
    temp_df = pd.DataFrame({
        'Probabilidad': probabilities,
        'Clase Real': y_true
    })
    
    # Graficar histograma
    sns.histplot(data=temp_df, x='Probabilidad', hue='Clase Real', bins=20, 
                 multiple='stack', palette=['green', 'red'])
    plt.axvline(x=0.5, color='black', linestyle='--')
    plt.title('Distribución de Probabilidades Predichas')
    plt.xlabel('Probabilidad de ser Maligno')
    plt.ylabel('Frecuencia')
    plt.legend(['Umbral (0.5)', 'Benigno (0)', 'Maligno (1)'])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_misclassified_examples(probabilities, y_true, y_pred, figsize=(10, 6), save_path=None):
    """
    Visualiza las probabilidades de los ejemplos mal clasificados.
    
    Args:
        probabilities: Probabilidades predichas (valores continuos entre 0 y 1)
        y_true: Etiquetas verdaderas (0 o 1)
        y_pred: Etiquetas predichas (0 o 1)
        figsize: Tamaño de la figura (ancho, alto)
        save_path: Ruta para guardar la imagen (opcional)
    """
    # Aplanar los arrays para asegurarse de que son unidimensionales
    probabilities = probabilities.flatten()
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Identificar ejemplos mal clasificados
    misclassified = (y_true != y_pred)
    
    if np.sum(misclassified) > 0:
        plt.figure(figsize=figsize)
        
        # Usar índices de los ejemplos mal clasificados
        indices = np.where(misclassified)[0]
        
        # Graficar
        plt.scatter(indices, probabilities[misclassified], 
                    c=y_true[misclassified], cmap='coolwarm', alpha=0.7)
        plt.axhline(y=0.5, color='black', linestyle='--')
        plt.title('Probabilidades de Ejemplos Mal Clasificados')
        plt.xlabel('Índice de Ejemplo')
        plt.ylabel('Probabilidad')
        plt.colorbar(label='Clase Real')
        plt.legend(['Umbral (0.5)'])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    else:
        print("No hay ejemplos mal clasificados para visualizar.")

def evaluate_binary_classifier(y_true, y_pred_prob, threshold=0.5):
    """
    Evalúa un clasificador binario y retorna múltiples métricas.
    
    Args:
        y_true: Etiquetas verdaderas (0 o 1)
        y_pred_prob: Probabilidades predichas (valores continuos entre 0 y 1)
        threshold: Umbral para convertir probabilidades en clases (default: 0.5)
        
    Returns:
        dict: Diccionario con métricas (accuracy, precision, recall, f1_score)
    """
    # Convertir probabilidades a predicciones binarias
    y_pred = (y_pred_prob > threshold).astype(int)
    
    # Calcular accuracy manualmente
    accuracy = np.mean(y_pred.flatten() == y_true.flatten())
    
    # Calcular otras métricas
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Empaquetar resultados
    metrics = {
        'accuracy': accuracy,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'threshold': threshold,
        'n_samples': len(y_true),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    return metrics