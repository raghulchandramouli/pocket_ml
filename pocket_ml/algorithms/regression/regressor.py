import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso 
from sklearn.svm import SVR

class Regressor:
    """
    A Unified class for different regression algorithms.
    
    Parameters
    ----------
    model_type : str, optional
        The type of regression model to use. Default is 'linear'.
        Options: 'linear', 'ridge', 'lasso', 'svr', 'rf', 'gbm'.

    model_params : dict, optional
        The parameters for the regression model. Default is None.
    """
    
    def __init__(self, model_type='random_forest', **kwargs):
        self.model_type = model_type
        self.kwargs = kwargs
        self.model = self._get_model()
        
    def _get_model(self):
        
        """Init the specific Regressor model."""
        
        models = {
            'random_forest': RandomForestClassifier,
            'linear': LinearRegression,
            'svr': SVR,
            'ridge': Ridge,
            'lasso': Lasso,
            'gradient_boosting': GradientBoostingClassifier
        }
        
        if self.model_type not in models:
            raise ValueError(f"Invalid model type: {self.model_type}")
        
        return models[self.model_type](**self.kwargs)
    
    def fit(self, X, y):
        """
            Train the Regressor Model
            
            Params:
            X : array-like of shape (n_samples, n_features)
                The training input samples.
            y : array-like of shape (n_samples,) or (n_samples, n_targets)
                The target values.
        """
        
        return self.model.fit(X, y)
    
    def predict(self, X):
        """
        Make predictions for X
        
        Params:
        X : Array-like of shape (n_samples, n_features)
            The input samples.

        Returns:
        y_pred : array-like of shape (n_samples,)
            The predicted values.
        """
        
        return self.model.predict(X)
    
    def score(self, X, y):
        """
            Return the coefficient of determination of the prediction. R^2
            
            Params:
            X : array-like of shape (n_samples, n_features)
                Test samples.
            Y : array-like of shape (n_samples,) or (n_samples, n_targets)
                True values for X.
                
            Returns:
            score : float
                R^2 of self.predict(X) wrt. y.
        """
        
        return self.model.score(X, y)