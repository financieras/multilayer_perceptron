"""
Funciones para la evaluación de modelos de clasificación.
Este módulo contiene funciones para calcular diferentes métricas de rendimiento
y generar visualizaciones para evaluar modelos de clasificación.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import pandas as pd
from tabulate import tabulate

# Métricas de evaluación

def accuracy(y_true, y_pred):
    """
    Calcula la precisión (accuracy) del modelo.
    
    Parámetros:
    y_true (numpy.ndarray): Etiquetas verdaderas
    y_pred (numpy.ndarray): Etiquetas predichas
    
    Retorna:
    float: Precisión del modelo (número de predicciones correctas / número total de predicciones)
    """
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    """
    Calcula la precisión (precision) del modelo.
    Para clasificación binaria donde 1 es la clase positiva.
    
    Parámetros:
    y_true (numpy.ndarray): Etiquetas verdaderas
    y_pred (numpy.ndarray): Etiquetas predichas
    
    Retorna:
    float: Precisión del modelo (true positives / (true positives + false positives))
    """
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    
    if true_positives + false_positives == 0:
        return 0
    
    return true_positives / (true_positives + false_positives)

def recall(y_true, y_pred):
    """
    Calcula la exhaustividad (recall) del modelo.
    Para clasificación binaria donde 1 es la clase positiva.
    
    Parámetros:
    y_true (numpy.ndarray): Etiquetas verdaderas
    y_pred (numpy.ndarray): Etiquetas predichas
    
    Retorna:
    float: Exhaustividad del modelo (true positives / (true positives + false negatives))
    """
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    
    if true_positives + false_negatives == 0:
        return 0
    
    return true_positives / (true_positives + false_negatives)

def f1_score(y_true, y_pred):
    """
    Calcula el F1-Score del modelo.
    Para clasificación binaria donde 1 es la clase positiva.
    
    Parámetros:
    y_true (numpy.ndarray): Etiquetas verdaderas
    y_pred (numpy.ndarray): Etiquetas predichas
    
    Retorna:
    float: F1-Score del modelo (2 * (precision * recall) / (precision + recall))
    """
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    
    if prec + rec == 0:
        return 0
    
    return 2 * (prec * rec) / (prec + rec)

def confusion_matrix(y_true, y_pred, normalize=False):
    """
    Calcula la matriz de confusión del modelo.
    
    Parámetros:
    y_true (numpy.ndarray): Etiquetas verdaderas
    y_pred (numpy.ndarray): Etiquetas predichas
    normalize (bool): Si se normaliza la matriz por filas
    
    Retorna:
    numpy.ndarray: Matriz de confusión
    """
    n_classes = max(np.max(y_true), np.max(y_pred)) + 1
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    for i in range(len(y_true)):
        conf_matrix[y_true[i], y_pred[i]] += 1
    
    if normalize:
        row_sums = conf_matrix.sum(axis=1, keepdims=True)
        conf_matrix = conf_matrix.astype('float') / row_sums
    
    return conf_matrix

def calculate_metrics(y_true, y_pred, class_names=None):
    """
    Calcula varias métricas de rendimiento para un modelo de clasificación.
    
    Parámetros:
    y_true (numpy.ndarray): Etiquetas verdaderas
    y_pred (numpy.ndarray): Etiquetas predichas
    class_names (list): Nombres de las clases
    
    Retorna:
    dict: Diccionario con las métricas calculadas
    """
    if len(y_true.shape) > 1:
        y_true = y_true.flatten()
    
    if len(y_pred.shape) > 1:
        y_pred = y_pred.flatten()
    
    if not class_names:
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        class_names = [f"Clase {i}" for i in unique_classes]
    
    metrics = {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "class_names": class_names
    }
    
    return metrics

# Visualizaciones

def plot_confusion_matrix(conf_matrix, class_names=None, normalize=False, title="Matriz de Confusión",
                          cmap=plt.cm.Blues, figsize=(8, 6), output_path=None):
    """
    Visualiza la matriz de confusión.
    
    Parámetros:
    conf_matrix (numpy.ndarray): Matriz de confusión
    class_names (list): Nombres de las clases
    normalize (bool): Si se normaliza la matriz por filas
    title (str): Título del gráfico
    cmap (matplotlib.colors.Colormap): Mapa de colores
    figsize (tuple): Tamaño de la figura
    output_path (str): Ruta donde guardar la figura
    
    Retorna:
    matplotlib.figure.Figure: Figura generada
    """
    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1, keepdims=True)
        fmt = '.2f'
    else:
        fmt = 'd'
    
    if class_names is None:
        class_names = [f"Clase {i}" for i in range(conf_matrix.shape[0])]
    
    plt.figure(figsize=figsize)
    plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Agregar valores en las celdas
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], fmt),
                     ha="center", va="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('Etiqueta verdadera')
    plt.xlabel('Etiqueta predicha')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def plot_roc_curve(y_true, y_prob, figsize=(8, 6), output_path=None):
    """
    Visualiza la curva ROC (Receiver Operating Characteristic).
    
    Parámetros:
    y_true (numpy.ndarray): Etiquetas verdaderas
    y_prob (numpy.ndarray): Probabilidades predichas para la clase positiva
    figsize (tuple): Tamaño de la figura
    output_path (str): Ruta donde guardar la figura
    
    Retorna:
    tuple: (matplotlib.figure.Figure, float) - Figura generada y área bajo la curva
    """
    # Aplanar arrays si es necesario
    if len(y_true.shape) > 1:
        y_true = y_true.flatten()
    
    if len(y_prob.shape) > 1:
        y_prob = y_prob.flatten()
    
    # Ordenar por probabilidad y calcular tasas
    indices = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[indices]
    
    n_positive = np.sum(y_true == 1)
    n_negative = len(y_true) - n_positive
    
    tpr = np.zeros(len(y_true) + 1)
    fpr = np.zeros(len(y_true) + 1)
    
    tp = 0
    fp = 0
    
    for i in range(len(y_true)):
        if y_true_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        
        tpr[i+1] = tp / n_positive if n_positive > 0 else 0
        fpr[i+1] = fp / n_negative if n_negative > 0 else 0
    
    # Calcular AUC
    auc = np.trapz(tpr, fpr)
    
    # Crear figura
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, 'b-', label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'r--', label='Azar')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf(), auc

def plot_learning_curves(history, figsize=(12, 5), output_path=None):
    """
    Visualiza las curvas de aprendizaje durante el entrenamiento.
    
    Parámetros:
    history (dict): Historial de entrenamiento
    figsize (tuple): Tamaño de la figura
    output_path (str): Ruta donde guardar la figura
    
    Retorna:
    matplotlib.figure.Figure: Figura generada
    """
    plt.figure(figsize=figsize)
    
    # Gráfica de pérdida
    plt.subplot(1, 2, 1)
    plt.plot(history["training_loss"], label="Entrenamiento")
    if "validation_loss" in history and history["validation_loss"]:
        plt.plot(history["validation_loss"], label="Validación")
    plt.title("Función de pérdida")
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.legend()
    plt.grid(True)
    
    # Gráfica de precisión
    plt.subplot(1, 2, 2)
    plt.plot(history["training_accuracy"], label="Entrenamiento")
    if "validation_accuracy" in history and history["validation_accuracy"]:
        plt.plot(history["validation_accuracy"], label="Validación")
    plt.title("Precisión")
    plt.xlabel("Época")
    plt.ylabel("Precisión")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def generate_classification_report(y_true, y_pred, y_prob=None, class_names=None, 
                                   output_dir=None, model_name="model"):
    """
    Genera un informe completo de clasificación con métricas y visualizaciones.
    
    Parámetros:
    y_true (numpy.ndarray): Etiquetas verdaderas
    y_pred (numpy.ndarray): Etiquetas predichas
    y_prob (numpy.ndarray): Probabilidades predichas (opcional)
    class_names (list): Nombres de las clases
    output_dir (str): Directorio donde guardar los resultados
    model_name (str): Nombre del modelo
    
    Retorna:
    dict: Métricas y rutas a las visualizaciones generadas
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {}
    
    # Calcular métricas
    metrics = calculate_metrics(y_true, y_pred, class_names)
    report.update(metrics)
    
    # Si se proporcionan probabilidades, calcular AUC
    if y_prob is not None:
        _, auc = plot_roc_curve(y_true, y_prob)
        report["auc"] = auc
    
    # Guardar resultados si se proporciona un directorio de salida
    if output_dir:
        # Crear directorio para las figuras
        figures_dir = os.path.join(output_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        
        # Guardar matriz de confusión
        conf_matrix_path = os.path.join(figures_dir, f"{model_name}_confusion_matrix_{timestamp}.png")
        plot_confusion_matrix(
            np.array(metrics["confusion_matrix"]), 
            class_names=metrics["class_names"],
            output_path=conf_matrix_path
        )
        report["confusion_matrix_path"] = conf_matrix_path
        
        # Guardar curva ROC si se proporcionan probabilidades
        if y_prob is not None:
            roc_curve_path = os.path.join(figures_dir, f"{model_name}_roc_curve_{timestamp}.png")
            plot_roc_curve(y_true, y_prob, output_path=roc_curve_path)
            report["roc_curve_path"] = roc_curve_path
        
        # Guardar métricas en formato JSON
        metrics_path = os.path.join(output_dir, f"{model_name}_metrics_{timestamp}.json")
        
        # Eliminar la matriz de confusión del reporte JSON (ya que se guarda como imagen)
        json_report = report.copy()
        json_report["confusion_matrix"] = "Saved as image"
        
        with open(metrics_path, "w") as f:
            json.dump(json_report, f, indent=4)
        
        report["metrics_path"] = metrics_path
    
    return report

def print_classification_report(metrics, decimals=4):
    """
    Imprime un informe de clasificación formateado en la consola.
    
    Parámetros:
    metrics (dict): Métricas de clasificación
    decimals (int): Número de decimales a mostrar
    """
    print("\n" + "=" * 50)
    print(" " * 15 + "INFORME DE CLASIFICACIÓN")
    print("=" * 50)
    
    print(f"\nPrecisión (Accuracy): {metrics['accuracy']:.{decimals}f}")
    print(f"Precisión (Precision): {metrics['precision']:.{decimals}f}")
    print(f"Exhaustividad (Recall): {metrics['recall']:.{decimals}f}")
    print(f"F1-Score: {metrics['f1_score']:.{decimals}f}")
    
    if "auc" in metrics:
        print(f"Área bajo la curva ROC (AUC): {metrics['auc']:.{decimals}f}")
    
    # Imprimir matriz de confusión
    print("\nMatriz de Confusión:")
    conf_matrix = np.array(metrics["confusion_matrix"])
    class_names = metrics.get("class_names", [f"Clase {i}" for i in range(conf_matrix.shape[0])])
    
    headers = [""] + class_names
    table = []
    
    for i, row in enumerate(conf_matrix):
        table.append([class_names[i]] + list(row))
    
    print(tabulate(table, headers=headers, tablefmt="grid"))
    
    print("\n" + "=" * 50)

def compare_models(models_metrics, model_names=None, figsize=(10, 8), output_path=None):
    """
    Compara varios modelos basándose en sus métricas.
    
    Parámetros:
    models_metrics (list): Lista de diccionarios con las métricas de cada modelo
    model_names (list): Nombres de los modelos
    figsize (tuple): Tamaño de la figura
    output_path (str): Ruta donde guardar la figura
    
    Retorna:
    matplotlib.figure.Figure: Figura generada
    """
    if model_names is None:
        model_names = [f"Modelo {i+1}" for i in range(len(models_metrics))]
    
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    if "auc" in models_metrics[0]:
        metrics.append("auc")
    
    # Crear DataFrame con las métricas
    data = []
    for name, metrics_dict in zip(model_names, models_metrics):
        row = {"Modelo": name}
        for metric in metrics:
            row[metric] = metrics_dict[metric]
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Crear gráfica de barras
    plt.figure(figsize=figsize)
    
    x = np.arange(len(model_names))
    width = 0.15
    multiplier = 0
    
    for metric in metrics:
        offset = width * multiplier
        plt.bar(x + offset, df[metric], width, label=metric)
        multiplier += 1
    
    plt.xlabel('Modelos')
    plt.ylabel('Valor')
    plt.title('Comparación de modelos por métricas')
    plt.xticks(x + width * (len(metrics) - 1) / 2, model_names, rotation=45, ha='right')
    plt.ylim(0, 1.05)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(metrics))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def plot_feature_importance(feature_importances, feature_names=None, top_n=None, 
                           figsize=(10, 8), output_path=None):
    """
    Visualiza la importancia de las características.
    
    Parámetros:
    feature_importances (numpy.ndarray): Valores de importancia
    feature_names (list): Nombres de las características
    top_n (int): Número de características principales a mostrar
    figsize (tuple): Tamaño de la figura
    output_path (str): Ruta donde guardar la figura
    
    Retorna:
    matplotlib.figure.Figure: Figura generada
    """
    if feature_names is None:
        feature_names = [f"Característica {i+1}" for i in range(len(feature_importances))]
    
    # Ordenar características por importancia
    indices = np.argsort(feature_importances)[::-1]
    
    if top_n is not None:
        indices = indices[:top_n]
    
    sorted_importances = feature_importances[indices]
    sorted_names = [feature_names[i] for i in indices]
    
    plt.figure(figsize=figsize)
    plt.bar(range(len(sorted_importances)), sorted_importances)
    plt.xticks(range(len(sorted_importances)), sorted_names, rotation=90)
    plt.title("Importancia de características")
    plt.xlabel("Características")
    plt.ylabel("Importancia")
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def plot_decision_regions(X, y, model, feature_indices=(0, 1), mesh_step_size=0.02, 
                         figsize=(10, 8), output_path=None):
    """
    Visualiza las regiones de decisión de un modelo para dos características seleccionadas.
    
    Parámetros:
    X (numpy.ndarray): Datos de características
    y (numpy.ndarray): Etiquetas
    model: Modelo entrenado con método predict
    feature_indices (tuple): Índices de las dos características a visualizar
    mesh_step_size (float): Tamaño del paso para la malla
    figsize (tuple): Tamaño de la figura
    output_path (str): Ruta donde guardar la figura
    
    Retorna:
    matplotlib.figure.Figure: Figura generada
    """
    # Seleccionar solo dos características
    X_selected = X[:, feature_indices]
    
    # Definir los límites de la malla
    x_min, x_max = X_selected[:, 0].min() - 1, X_selected[:, 0].max() + 1
    y_min, y_max = X_selected[:, 1].min() - 1, X_selected[:, 1].max() + 1
    
    # Crear malla
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size),
                         np.arange(y_min, y_max, mesh_step_size))
    
    # Crear datos para predicción
    mesh_data = np.c_[xx.ravel(), yy.ravel()]
    
    # Si el modelo espera más características, añadir ceros para las no visualizadas
    if X.shape[1] > 2:
        # Crear una matriz de ceros con la forma adecuada
        full_mesh_data = np.zeros((mesh_data.shape[0], X.shape[1]))
        
        # Asignar las dos características seleccionadas
        full_mesh_data[:, feature_indices[0]] = mesh_data[:, 0]
        full_mesh_data[:, feature_indices[1]] = mesh_data[:, 1]
        
        # Usar los datos completos para la predicción
        Z = model.predict(full_mesh_data)
    else:
        # Si solo hay dos características, usar directamente mesh_data
        Z = model.predict(mesh_data)
    
    # Convertir a valores binarios si es necesario
    if len(Z.shape) > 1 and Z.shape[1] == 1:
        Z = (Z > 0.5).astype(int).flatten()
    
    # Reshape para la visualización
    Z = Z.reshape(xx.shape)
    
    # Crear figura
    plt.figure(figsize=figsize)
    
    # Visualizar regiones de decisión
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
    
    # Visualizar puntos de datos
    for class_value in np.unique(y):
        plt.scatter(X_selected[y == class_value, 0], 
                   X_selected[y == class_value, 1],
                   alpha=0.8,
                   label=f'Clase {class_value}')
    
    plt.xlabel(f'Característica {feature_indices[0]}')
    plt.ylabel(f'Característica {feature_indices[1]}')
    plt.title('Regiones de decisión')
    plt.legend()
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

# Funciones para análisis de resultados experimentales

def analyze_hyperparameter_experiments(experiments_results, figsize=(12, 10), output_path=None):
    """
    Analiza los resultados de experimentos con diferentes hiperparámetros.
    
    Parámetros:
    experiments_results (list): Lista de diccionarios con los resultados de cada experimento
    figsize (tuple): Tamaño de la figura
    output_path (str): Ruta donde guardar la figura
    
    Retorna:
    tuple: (pd.DataFrame, matplotlib.figure.Figure) - DataFrame y figura con el análisis
    """
    # Convertir a DataFrame
    df = pd.DataFrame(experiments_results)
    
    # Encontrar el mejor experimento
    best_idx = df['accuracy'].idxmax()
    best_config = df.iloc[best_idx].to_dict()
    
    # Crear figura con subgráficos
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Análisis de Hiperparámetros', fontsize=16)
    
    # Parámetros a analizar
    params = ['hidden_layers', 'neurons', 'learning_rate']
    metrics = ['accuracy', 'loss']
    
    # Para cada parámetro, mostrar su efecto en la precisión
    for i, param in enumerate(params):
        ax = axes[i // 2, i % 2]
        
        grouped = df.groupby(param)[metrics].mean().reset_index()
        
        # Barras para accuracy
        ax.bar(grouped[param].astype(str), grouped['accuracy'], color='skyblue', alpha=0.7)
        ax.set_title(f'Efecto de {param}')
        ax.set_xlabel(param)
        ax.set_ylabel('Precisión')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Línea para loss
        ax2 = ax.twinx()
        ax2.plot(range(len(grouped)), grouped['loss'], 'r-', marker='o')
        ax2.set_ylabel('Pérdida', color='r')
        ax2.tick_params(axis='y', colors='r')
    
    # Tabla con los mejores resultados
    top_n = 5
    top_configs = df.nlargest(top_n, 'accuracy')
    
    axes[1, 1].axis('off')
    axes[1, 1].set_title(f'Top {top_n} Configuraciones')
    
    cell_text = []
    for i, row in top_configs.iterrows():
        cell_text.append([
            f"{row['hidden_layers']}",
            f"{row['neurons']}",
            f"{row['activation']}",
            f"{row['optimizer']}",
            f"{row['learning_rate']:.4f}",
            f"{row['accuracy']:.4f}"
        ])
    
    table = axes[1, 1].table(
        cellText=cell_text,
        colLabels=['Capas', 'Neuronas', 'Activación', 'Optimizador', 'LR', 'Precisión'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return df, fig

def export_model_summary(model, metrics, config, output_path):
    """
    Exporta un resumen del modelo a un archivo.
    
    Parámetros:
    model: Modelo entrenado
    metrics (dict): Métricas del modelo
    config (dict): Configuración del modelo
    output_path (str): Ruta donde guardar el resumen
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(" " * 15 + "RESUMEN DEL MODELO" + " " * 15 + "\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Fecha y hora: {timestamp}\n\n")
        
        # Información de la arquitectura
        f.write("ARQUITECTURA DEL MODELO\n")
        f.write("-" * 30 + "\n")
        f.write(f"Capas ocultas: {config.get('hidden_layers', 'No especificado')}\n")
        f.write(f"Neuronas por capa: {config.get('neurons_per_layer', 'No especificado')}\n")
        f.write(f"Función de activación: {config.get('activation', 'No especificado')}\n")
        f.write(f"Función de activación (salida): {config.get('output_activation', 'No especificado')}\n")
        f.write(f"Inicialización de pesos: {config.get('weights_initializer', 'No especificado')}\n")
        f.write(f"Función de pérdida: {config.get('loss', 'No especificado')}\n\n")
        
        # Información del entrenamiento
        f.write("PARÁMETROS DE ENTRENAMIENTO\n")
        f.write("-" * 30 + "\n")
        f.write(f"Optimizador: {config.get('optimizer', 'No especificado')}\n")
        f.write(f"Tasa de aprendizaje: {config.get('learning_rate', 'No especificado')}\n")
        f.write(f"Tamaño de lote: {config.get('batch_size', 'No especificado')}\n")
        f.write(f"Épocas: {config.get('epochs', 'No especificado')}\n")
        f.write(f"Early stopping: {config.get('early_stopping', 'No especificado')}\n\n")
        
        # Métricas de rendimiento
        f.write("MÉTRICAS DE RENDIMIENTO\n")
        f.write("-" * 30 + "\n")
        f.write(f"Precisión (Accuracy): {metrics.get('accuracy', 'No disponible'):.4f}\n")
        f.write(f"Precisión (Precision): {metrics.get('precision', 'No disponible'):.4f}\n")
        f.write(f"Exhaustividad (Recall): {metrics.get('recall', 'No disponible'):.4f}\n")
        f.write(f"F1-Score: {metrics.get('f1_score', 'No disponible'):.4f}\n")
        
        if "auc" in metrics:
            f.write(f"Área bajo la curva ROC (AUC): {metrics['auc']:.4f}\n")
        
        f.write("\n")
        
        # Información adicional
        f.write("INFORMACIÓN ADICIONAL\n")
        f.write("-" * 30 + "\n")
        f.write(f"Matriz de confusión: {metrics.get('confusion_matrix_path', 'No disponible')}\n")
        if "roc_curve_path" in metrics:
            f.write(f"Curva ROC: {metrics['roc_curve_path']}\n")