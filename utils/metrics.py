import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    cohen_kappa_score, 
    classification_report,
    confusion_matrix
)

def calculate_metrics(y_true, y_pred, class_names):
    """
    Calculate classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    
    Returns:
        Dictionary of metrics
    """
    # Overall Accuracy
    oa = accuracy_score(y_true, y_pred) * 100
    
    # Average Accuracy (per class)
    cm = confusion_matrix(y_true, y_pred)
    aa = np.mean(np.diag(cm) / np.sum(cm, axis=1)) * 100
    
    # Kappa coefficient
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # Per-class metrics
    report = classification_report(
        y_true, y_pred, 
        target_names=class_names[1:],  # Exclude background
        output_dict=True,
        zero_division=0
    )
    
    metrics = {
        'overall_accuracy': oa,
        'average_accuracy': aa,
        'kappa': kappa,
        'confusion_matrix': cm,
        'classification_report': report
    }
    
    return metrics

def print_metrics(metrics, class_names):
    """Print metrics in a formatted way"""
    print("\n" + "="*60)
    print("CLASSIFICATION RESULTS")
    print("="*60)
    print(f"Overall Accuracy (OA): {metrics['overall_accuracy']:.2f}%")
    print(f"Average Accuracy (AA): {metrics['average_accuracy']:.2f}%")
    print(f"Kappa Coefficient (κ): {metrics['kappa']:.4f}")
    print("="*60)
    
    print("\nPer-Class Accuracy:")
    print("-"*60)
    report = metrics['classification_report']
    for i, class_name in enumerate(class_names[1:], 1):
        if str(i-1) in report:
            acc = report[str(i-1)]['precision'] * 100
            print(f"{class_name:30s}: {acc:6.2f}%")
    print("="*60)