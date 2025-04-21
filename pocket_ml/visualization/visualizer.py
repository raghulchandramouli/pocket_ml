import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

class Visualizer:
    """A class for creating common machine learning visualizations."""
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, labels=None):
        """Plot confusion matrix.
        
        Parameters
        ----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        labels : list, optional
            List of labels to index the matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        return plt
    
    @staticmethod
    def plot_roc_curve(y_true, y_prob):
        """Plot ROC curve.
        
        Parameters
        ----------
        y_true : array-like
            True labels
        y_prob : array-like
            Predicted probabilities
        """
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        return plt
    
    @staticmethod
    def plot_feature_importance(model, feature_names):
        """Plot feature importance.
        
        Parameters
        ----------
        model : estimator object
            Fitted model with feature_importances_ attribute
        feature_names : list
            List of feature names
        """
        if not hasattr(model, 'feature_importances_'):
            raise AttributeError("Model doesn't have feature_importances_ attribute")
            
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importances')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), 
                   [feature_names[i] for i in indices], 
                   rotation=45)
        return plt