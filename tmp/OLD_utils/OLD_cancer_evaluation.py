"""
Funciones específicas para la evaluación del modelo en el conjunto de datos de cáncer de mama.
Este módulo complementa las funciones de evaluación generales con funciones específicas
para este tipo de datos y aplicación médica.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from datetime import datetime
from tabulate import tabulate

def compute_clinical_metrics(y_true, y_pred, threshold=0.5):
    """
    Calcula métricas específicas para aplicaciones clínicas.
    
    Parámetros:
    y_true (numpy.ndarray): Valores verdaderos (0: maligno, 1: benigno)
    y_pred (numpy.ndarray): Probabilidades predichas
    threshold (float): Umbral para clasificación binaria
    
    Retorna:
    dict: Métricas clínicas calculadas
    """
    # Convertir predicciones a clases usando el umbral
    y_pred_binary = (y_pred > threshold).astype(int)
    
    # Matriz de confusión
    TP = np.sum((y_true == 1) & (y_pred_binary == 1))  # Verdaderos positivos (benignos correctos)
    TN = np.sum((y_true == 0) & (y_pred_binary == 0))  # Verdaderos negativos (malignos correctos)
    FP = np.sum((y_true == 0) & (y_pred_binary == 1))  # Falsos positivos (malignos como benignos)
    FN = np.sum((y_true == 1) & (y_pred_binary == 0))  # Falsos negativos (benignos como malignos)
    
    # Métricas básicas
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    # Métricas clínicas
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0  # Recall, sensibilidad
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0  # Especificidad
    ppv = TP / (TP + FP) if (TP + FP) > 0 else 0  # Valor predictivo positivo (precisión)
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0  # Valor predictivo negativo
    
    # F1-score y otras métricas
    f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
    
    # Métricas adicionales
    # Para cáncer: Falsos negativos son más peligrosos (un cáncer no detectado)
    false_negative_rate = FN / (FN + TP) if (FN + TP) > 0 else 0  # Tasa de falsos negativos
    false_positive_rate = FP / (FP + TN) if (FP + TN) > 0 else 0  # Tasa de falsos positivos
    
    # Valor diagnóstico
    diagnostic_odds_ratio = (TP * TN) / (FP * FN) if (FP * FN) > 0 else float('inf')
    
    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,  # Recall
        "specificity": specificity,
        "precision": ppv,  # Valor predictivo positivo
        "npv": npv,  # Valor predictivo negativo
        "f1_score": f1,
        "false_negative_rate": false_negative_rate,
        "false_positive_rate": false_positive_rate,
        "diagnostic_odds_ratio": diagnostic_odds_ratio,
        "confusion_matrix": {
            "TP": int(TP),
            "TN": int(TN), 
            "FP": int(FP),
            "FN": int(FN)
        }
    }

def find_optimal_threshold(y_true, y_pred, metric='f1', thresholds=None):
    """
    Encuentra el umbral óptimo para maximizar una métrica específica.
    
    Parámetros:
    y_true (numpy.ndarray): Valores verdaderos
    y_pred (numpy.ndarray): Probabilidades predichas
    metric (str): Métrica a optimizar ('f1', 'sensitivity', 'specificity', 'balanced_accuracy')
    thresholds (numpy.ndarray): Lista de umbrales a probar (None para generar automáticamente)
    
    Retorna:
    tuple: (umbral óptimo, valor de la métrica, resultados para todos los umbrales)
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    
    results = []
    
    for threshold in thresholds:
        metrics = compute_clinical_metrics(y_true, y_pred, threshold)
        
        if metric == 'balanced_accuracy':
            value = (metrics['sensitivity'] + metrics['specificity']) / 2
        else:
            value = metrics.get(metric, 0)
        
        results.append({
            'threshold': threshold,
            'value': value,
            'metrics': metrics
        })
    
    # Encontrar el umbral óptimo
    best_result = max(results, key=lambda x: x['value'])
    
    return best_result['threshold'], best_result['value'], results

def plot_threshold_metrics(y_true, y_pred, metrics_to_plot=None, thresholds=None):
    """
    Visualiza cómo cambian las métricas con diferentes umbrales.
    
    Parámetros:
    y_true (numpy.ndarray): Valores verdaderos
    y_pred (numpy.ndarray): Probabilidades predichas
    metrics_to_plot (list): Lista de métricas a visualizar
    thresholds (numpy.ndarray): Lista de umbrales a probar (None para generar automáticamente)
    
    Retorna:
    matplotlib.figure.Figure: Figura generada
    """
    if metrics_to_plot is None:
        metrics_to_plot = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score']
    
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    
    # Calcular métricas para cada umbral
    threshold_results = []
    for threshold in thresholds:
        metrics = compute_clinical_metrics(y_true, y_pred, threshold)
        threshold_results.append(metrics)
    
    # Crear gráfico
    plt.figure(figsize=(12, 8))
    
    for metric in metrics_to_plot:
        values = [result[metric] for result in threshold_results]
        plt.plot(thresholds, values, marker='', linewidth=2, label=metric)
    
    plt.title('Variación de métricas según el umbral de decisión')
    plt.xlabel('Umbral')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

def plot_roc_with_optimal_threshold(y_true, y_pred, figsize=(10, 8), output_path=None):
    """
    Visualiza la curva ROC con el punto óptimo para maximizar diferentes métricas.
    
    Parámetros:
    y_true (numpy.ndarray): Valores verdaderos
    y_pred (numpy.ndarray): Probabilidades predichas
    figsize (tuple): Tamaño de la figura
    output_path (str): Ruta donde guardar la figura
    
    Retorna:
    tuple: (figura, resultados óptimos)
    """
    # Aplanar arrays si es necesario
    if len(y_true.shape) > 1:
        y_true = y_true.flatten()
    
    if len(y_pred.shape) > 1:
        y_pred = y_pred.flatten()
    
    # Calcular puntos de la curva ROC
    thresholds = np.linspace(0, 1, 100)
    tpr = []  # True Positive Rate (Sensitivity)
    fpr = []  # False Positive Rate (1 - Specificity)
    
    for threshold in thresholds:
        metrics = compute_clinical_metrics(y_true, y_pred, threshold)
        tpr.append(metrics['sensitivity'])
        fpr.append(metrics['false_positive_rate'])
    
    # Calcular AUC
    auc = np.trapz(tpr, fpr)
    
    # Encontrar umbrales óptimos para diferentes métricas
    optimal_thresholds = {
        'f1': find_optimal_threshold(y_true, y_pred, 'f1'),
        'balanced_accuracy': find_optimal_threshold(y_true, y_pred, 'balanced_accuracy'),
        'sensitivity': find_optimal_threshold(y_true, y_pred, 'sensitivity'),
        'specificity': find_optimal_threshold(y_true, y_pred, 'specificity')
    }
    
    # Obtener puntos óptimos en la curva ROC
    optimal_points = {}
    for metric, (threshold, value, _) in optimal_thresholds.items():
        metrics = compute_clinical_metrics(y_true, y_pred, threshold)
        optimal_points[metric] = (metrics['false_positive_rate'], metrics['sensitivity'])
    
    # Crear figura
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'r--', label='Azar')
    
    # Marcar puntos óptimos
    markers = {'f1': 'o', 'balanced_accuracy': 's', 'sensitivity': '^', 'specificity': 'D'}
    for metric, (x, y) in optimal_points.items():
        threshold = optimal_thresholds[metric][0]
        plt.plot(x, y, markers[metric], markersize=10, 
                label=f'Óptimo para {metric} (umbral={threshold:.2f})')
    
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
    plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
    plt.title('Curva ROC con puntos óptimos para diferentes métricas')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf(), optimal_thresholds

def plot_cancer_confusion_matrix(y_true, y_pred, threshold=0.5, 
                                class_names=['Maligno (0)', 'Benigno (1)'],
                                title="Matriz de Confusión para Diagnóstico de Cáncer",
                                cmap=plt.cm.Blues, figsize=(10, 8), output_path=None):
    """
    Visualiza la matriz de confusión con interpretación específica para el diagnóstico de cáncer.
    
    Parámetros:
    y_true (numpy.ndarray): Valores verdaderos
    y_pred (numpy.ndarray): Probabilidades predichas
    threshold (float): Umbral para clasificación binaria
    class_names (list): Nombres de las clases
    title (str): Título del gráfico
    cmap (matplotlib.colors.Colormap): Mapa de colores
    figsize (tuple): Tamaño de la figura
    output_path (str): Ruta donde guardar la figura
    
    Retorna:
    matplotlib.figure.Figure: Figura generada
    """
    # Convertir a binario
    y_pred_binary = (y_pred > threshold).astype(int)
    
    # Calcular matriz de confusión
    metrics = compute_clinical_metrics(y_true, y_pred_binary)
    cm = np.array([
        [metrics['confusion_matrix']['TN'], metrics['confusion_matrix']['FP']],
        [metrics['confusion_matrix']['FN'], metrics['confusion_matrix']['TP']]
    ])
    
    # Crear figura
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    # Etiquetas
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    
    # Agregar valores y explicaciones en las celdas
    thresh = cm.max() / 2
    explanations = [
        ['Verdadero Negativo\n(Maligno correcto)', 'Falso Positivo\n(Maligno como Benigno)'],
        ['Falso Negativo\n(Benigno como Maligno)', 'Verdadero Positivo\n(Benigno correcto)']
    ]
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]}\n{explanations[i][j]}",
                     ha="center", va="center", fontsize=9,
                     color="white" if cm[i, j] > thresh else "black")
    
    # Agregar métricas clave
    plt.text(1.5, -0.3, f"Precisión: {metrics['accuracy']:.4f}", fontsize=10, ha="center")
    plt.text(1.5, -0.2, f"Sensibilidad: {metrics['sensitivity']:.4f}", fontsize=10, ha="center")
    plt.text(1.5, -0.1, f"Especificidad: {metrics['specificity']:.4f}", fontsize=10, ha="center")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def generate_clinical_report(y_true, y_pred, output_dir=None, prefix="cancer_model"):
    """
    Genera un informe clínico completo del modelo.
    
    Parámetros:
    y_true (numpy.ndarray): Valores verdaderos
    y_pred (numpy.ndarray): Probabilidades predichas
    output_dir (str): Directorio donde guardar los resultados
    prefix (str): Prefijo para los archivos generados
    
    Retorna:
    dict: Informe con resultados y métricas
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {}
    
    # Calcular métricas con umbral por defecto
    default_metrics = compute_clinical_metrics(y_true, y_pred)
    report['default_metrics'] = default_metrics
    
    # Encontrar umbrales óptimos
    f1_threshold, f1_value, _ = find_optimal_threshold(y_true, y_pred, 'f1')
    report['optimal_thresholds'] = {
        'f1': {'threshold': f1_threshold, 'value': f1_value},
        'balanced_accuracy': find_optimal_threshold(y_true, y_pred, 'balanced_accuracy')[:2],
        'sensitivity': find_optimal_threshold(y_true, y_pred, 'sensitivity')[:2],
        'specificity': find_optimal_threshold(y_true, y_pred, 'specificity')[:2]
    }
    
    # Calcular métricas con umbral óptimo para F1
    optimal_metrics = compute_clinical_metrics(y_true, y_pred, f1_threshold)
    report['optimal_metrics'] = optimal_metrics
    
    # Guardar resultados si se proporciona un directorio de salida
    if output_dir:
        # Crear directorios necesarios
        figures_dir = os.path.join(output_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        
        # Generar y guardar curva ROC
        roc_path = os.path.join(figures_dir, f"{prefix}_roc_{timestamp}.png")
        _, optimal_thresholds = plot_roc_with_optimal_threshold(y_true, y_pred, output_path=roc_path)
        report['roc_path'] = roc_path
        
        # Generar y guardar matriz de confusión (umbral por defecto)
        cm_default_path = os.path.join(figures_dir, f"{prefix}_confusion_matrix_default_{timestamp}.png")
        plot_cancer_confusion_matrix(y_true, y_pred, threshold=0.5, output_path=cm_default_path)
        report['confusion_matrix_default_path'] = cm_default_path
        
        # Generar y guardar matriz de confusión (umbral óptimo)
        cm_optimal_path = os.path.join(figures_dir, f"{prefix}_confusion_matrix_optimal_{timestamp}.png")
        plot_cancer_confusion_matrix(y_true, y_pred, threshold=f1_threshold, output_path=cm_optimal_path)
        report['confusion_matrix_optimal_path'] = cm_optimal_path
        
        # Generar y guardar gráfico de métricas por umbral
        threshold_metrics_path = os.path.join(figures_dir, f"{prefix}_threshold_metrics_{timestamp}.png")
        plot_threshold_metrics(y_true, y_pred, output_path=threshold_metrics_path)
        report['threshold_metrics_path'] = threshold_metrics_path
        
        # Guardar métricas en formato JSON
        metrics_path = os.path.join(output_dir, f"{prefix}_clinical_metrics_{timestamp}.json")
        with open(metrics_path, "w") as f:
            json.dump(report, f, indent=4)
        report['metrics_path'] = metrics_path
        
        # Generar informe en formato texto
        report_path = os.path.join(output_dir, f"{prefix}_clinical_report_{timestamp}.txt")
        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write(" " * 25 + "INFORME CLÍNICO DEL MODELO" + " " * 25 + "\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("MÉTRICAS CON UMBRAL ESTÁNDAR (0.5)\n")
            f.write("-" * 40 + "\n")
            for metric, value in default_metrics.items():
                if metric != "confusion_matrix":
                    f.write(f"{metric.capitalize()}: {value:.4f}\n")
            
            f.write("\nMATRIZ DE CONFUSIÓN\n")
            f.write("-" * 40 + "\n")
            cm = default_metrics['confusion_matrix']
            f.write(f"Verdaderos Positivos (TP): {cm['TP']}\n")
            f.write(f"Verdaderos Negativos (TN): {cm['TN']}\n")
            f.write(f"Falsos Positivos (FP): {cm['FP']}\n")
            f.write(f"Falsos Negativos (FN): {cm['FN']}\n\n")
            
            f.write("UMBRALES ÓPTIMOS\n")
            f.write("-" * 40 + "\n")
            for metric, data in report['optimal_thresholds'].items():
                if isinstance(data, dict):
                    f.write(f"Para {metric}: {data['threshold']:.4f} (valor: {data['value']:.4f})\n")
                else:
                    threshold, value = data
                    f.write(f"Para {metric}: {threshold:.4f} (valor: {value:.4f})\n")
            
            f.write("\nMÉTRICAS CON UMBRAL ÓPTIMO PARA F1 ({:.4f})\n".format(f1_threshold))
            f.write("-" * 40 + "\n")
            for metric, value in optimal_metrics.items():
                if metric != "confusion_matrix":
                    f.write(f"{metric.capitalize()}: {value:.4f}\n")
            
            f.write("\nINTERPRETACIÓN CLÍNICA\n")
            f.write("-" * 40 + "\n")
            
            # Interpretación de la sensibilidad y especificidad
            sens = optimal_metrics['sensitivity']
            spec = optimal_metrics['specificity']
            f.write(f"Sensibilidad: {sens:.4f} - El modelo identifica correctamente el {sens*100:.1f}% de los casos benignos.\n")
            f.write(f"Especificidad: {spec:.4f} - El modelo identifica correctamente el {spec*100:.1f}% de los casos malignos.\n\n")
            
            # Interpretación de los valores predictivos
            ppv = optimal_metrics['precision']
            npv = optimal_metrics['npv']
            f.write(f"Valor predictivo positivo: {ppv:.4f} - Cuando el modelo predice 'benigno', acierta el {ppv*100:.1f}% de las veces.\n")
            f.write(f"Valor predictivo negativo: {npv:.4f} - Cuando el modelo predice 'maligno', acierta el {npv*100:.1f}% de las veces.\n\n")
            
            # Interpretación de las tasas de error
            fnr = optimal_metrics['false_negative_rate']
            fpr = optimal_metrics['false_positive_rate']
            f.write(f"Tasa de falsos negativos: {fnr:.4f} - El {fnr*100:.1f}% de los casos benignos son incorrectamente clasificados como malignos.\n")
            f.write(f"Tasa de falsos positivos: {fpr:.4f} - El {fpr*100:.1f}% de los casos malignos son incorrectamente clasificados como benignos.\n\n")
            
            # Interpretación del valor diagnóstico
            dor = optimal_metrics['diagnostic_odds_ratio']
            if np.isinf(dor):
                f.write("Odds ratio diagnóstico: Infinito - El modelo tiene una capacidad diagnóstica perfecta (0 falsos).\n")
            else:
                f.write(f"Odds ratio diagnóstico: {dor:.2f} - Un valor mayor indica mejor capacidad diagnóstica.\n")
        
        report['report_path'] = report_path
    
    return report

def print_clinical_report(metrics, decimals=4):
    """
    Imprime un informe clínico formateado en la consola.
    
    Parámetros:
    metrics (dict): Métricas clínicas
    decimals (int): Número de decimales a mostrar
    """
    print("\n" + "=" * 60)
    print(" " * 15 + "INFORME CLÍNICO DEL MODELO")
    print("=" * 60)
    
    print("\nMÉTRICAS PRINCIPALES:")
    print(f"Precisión (Accuracy): {metrics['accuracy']:.{decimals}f}")
    print(f"Sensibilidad (Sensitivity): {metrics['sensitivity']:.{decimals}f}")
    print(f"Especificidad (Specificity): {metrics['specificity']:.{decimals}f}")
    print(f"Valor Predictivo Positivo (Precision): {metrics['precision']:.{decimals}f}")
    print(f"Valor Predictivo Negativo: {metrics['npv']:.{decimals}f}")
    print(f"F1-Score: {metrics['f1_score']:.{decimals}f}")
    
    print("\nTASAS DE ERROR:")
    print(f"Tasa de Falsos Negativos: {metrics['false_negative_rate']:.{decimals}f}")
    print(f"Tasa de Falsos Positivos: {metrics['false_positive_rate']:.{decimals}f}")
    
    print("\nMATRIZ DE CONFUSIÓN:")
    cm = metrics['confusion_matrix']
    table = [
        ["", "Pred: Maligno (0)", "Pred: Benigno (1)"],
        ["Real: Maligno (0)", cm["TN"], cm["FP"]],
        ["Real: Benigno (1)", cm["FN"], cm["TP"]]
    ]
    print(tabulate(table, headers="firstrow", tablefmt="grid"))
    
    # Interpretación clínica
    print("\nINTERPRETACIÓN CLÍNICA:")
    sens = metrics['sensitivity']
    spec = metrics['specificity']
    print(f"- El modelo identifica correctamente el {sens*100:.1f}% de los casos benignos.")
    print(f"- El modelo identifica correctamente el {spec*100:.1f}% de los casos malignos.")
    
    # Valores predictivos
    ppv = metrics['precision']
    npv = metrics['npv']
    print(f"- Cuando el modelo predice 'benigno', acierta el {ppv*100:.1f}% de las veces.")
    print(f"- Cuando el modelo predice 'maligno', acierta el {npv*100:.1f}% de las veces.")
    
    print("\n" + "=" * 60)

def analyze_feature_importance_clinical(feature_importances, feature_names, top_n=10, 
                                      figsize=(12, 8), output_path=None):
    """
    Analiza y visualiza la importancia de las características con interpretación clínica.
    
    Parámetros:
    feature_importances (numpy.ndarray): Valores de importancia de características
    feature_names (list): Nombres de las características
    top_n (int): Número de características principales a mostrar
    figsize (tuple): Tamaño de la figura
    output_path (str): Ruta donde guardar la figura
    
    Retorna:
    tuple: (matplotlib.figure.Figure, pandas.DataFrame) - Figura y DataFrame con importancias
    """
    # Crear DataFrame con importancias
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })
    
    # Ordenar por importancia
    importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
    
    # Seleccionar top N características
    top_features = importance_df.head(top_n)
    
    # Crear visualización
    plt.figure(figsize=figsize)
    plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
    plt.xlabel('Importancia Relativa')
    plt.ylabel('Característica')
    plt.title(f'Top {top_n} Características Más Importantes para el Diagnóstico')
    plt.gca().invert_yaxis()  # Para que la más importante aparezca arriba
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf(), importance_df

def plot_threshold_metrics(y_true, y_pred, metrics_to_plot=None, thresholds=None, output_path=None):
    """
    Visualiza cómo cambian las métricas con diferentes umbrales.
    
    Parámetros:
    y_true (numpy.ndarray): Valores verdaderos
    y_pred (numpy.ndarray): Probabilidades predichas
    metrics_to_plot (list): Lista de métricas a visualizar
    thresholds (numpy.ndarray): Lista de umbrales a probar
    output_path (str): Ruta donde guardar la figura
    
    Retorna:
    matplotlib.figure.Figure: Figura generada
    """
    if metrics_to_plot is None:
        metrics_to_plot = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score']
    
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    
    # Calcular métricas para cada umbral
    metrics_by_threshold = {metric: [] for metric in metrics_to_plot}
    
    for threshold in thresholds:
        metrics = compute_clinical_metrics(y_true, y_pred, threshold)
        for metric in metrics_to_plot:
            metrics_by_threshold[metric].append(metrics[metric])
    
    # Crear gráfico
    plt.figure(figsize=(12, 8))
    
    for metric in metrics_to_plot:
        plt.plot(thresholds, metrics_by_threshold[metric], marker='', linewidth=2, label=metric)
    
    # Encontrar y marcar umbrales óptimos
    for metric in ['f1_score', 'accuracy']:
        if metric in metrics_to_plot:
            # Encontrar el umbral óptimo para esta métrica
            best_idx = np.argmax(metrics_by_threshold[metric])
            best_threshold = thresholds[best_idx]
            best_value = metrics_by_threshold[metric][best_idx]
            
            # Marcar en el gráfico
            plt.plot(best_threshold, best_value, 'o', markersize=8, 
                     label=f'Mejor {metric}: {best_value:.4f} (umbral={best_threshold:.2f})')
    
    plt.title('Variación de Métricas según el Umbral de Decisión')
    plt.xlabel('Umbral')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()