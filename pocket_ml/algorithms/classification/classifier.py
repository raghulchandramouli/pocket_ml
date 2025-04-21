import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

class Classifier:
    """A unified interface for different classification algorithms.
    
    Parameters
    ----------
    model_type : str
        Type of the classifier ('random_forest', 'logistic', 'svm', 'gradient_boosting')
    **kwargs : dict
        Additional parameters to pass to the classifier
    """
    
    def __init__(self, model_type='random_forest', **kwargs):
        self.model_type = model_type.lower()
        self.kwargs = kwargs
        self.model = self._get_model()
        
    def _get_model(self):
        """Initialize the specified classifier."""
        models = {
            'random_forest': RandomForestClassifier,
            'logistic': LogisticRegression,
            'svm': SVC,
            'gradient_boosting': GradientBoostingClassifier
        }
        
        if self.model_type not in models:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        return models[self.model_type](**self.kwargs)
    
    def fit(self, X, y):
        """Train the classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        """
        return self.model.fit(X, y)
    
    def predict(self, X):
        """Make predictions for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns
        -------
        array-like
            Predicted class labels
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples
            
        Returns
        -------
        array-like of shape (n_samples, n_classes)
            Class probabilities
        """
        if not hasattr(self.model, 'predict_proba'):
            raise AttributeError(f"{self.model_type} does not support probability predictions")
        return self.model.predict_proba(X)