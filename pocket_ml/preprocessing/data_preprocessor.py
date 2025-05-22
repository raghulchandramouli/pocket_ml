from numba import njit
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


# GPU Accelerated Functions
@njit
def _gpu_scale_features(X):
    """Scale the features of the input data.

    Parameters
    ----------
    X : array-like
        Input data

    Returns
    -------
    array-like
        Scaled data
    """
    return (X - X.mean(axis=0)) / X.std(axis=0)
class DataPreprocessor:
    """A unified interface for data preprocessing tasks.
    
    This class handles common preprocessing tasks including:
    - Missing value imputation
    - Feature scaling
    - Categorical encoding
    """
    
    def __init__(self, scaling=True, categorical_encoding=True, imputation=True):
        self.scaling = scaling
        self.categorical_encoding = categorical_encoding
        self.imputation = imputation
        
        self.scaler = StandardScaler() if scaling else None
        self.label_encoders = {} if categorical_encoding else None
        self.imputer = SimpleImputer(strategy='mean') if imputation else None
        
    def fit(self, X):
        """Fit the preprocessor to the data.
        
        Parameters
        ----------
        X : array-like
            Training data
        """
        if self.imputation:
            self.imputer.fit(X)
            
        if self.scaling:
            self.scaler.fit(X)
            
        if self.categorical_encoding:
            for col in X.select_dtypes(include=['object']).columns:
                self.label_encoders[col] = LabelEncoder().fit(X[col])
                
        return self
    
    def transform(self, X):
        """Apply the preprocessing transformations to the data.
        
        Parameters
        ----------
        X : array-like
            Data to transform
            
        Returns
        -------
        array-like
            Transformed data
        """
        def transform(self, X):
            if self.scaling:
                X = _gpu_scale_features(np.asarray(X))
        X_transformed = X.copy()
        
        if self.imputation:
            X_transformed = self.imputer.transform(X_transformed)
            
        if self.categorical_encoding:
            for col, encoder in self.label_encoders.items():
                X_transformed[col] = encoder.transform(X_transformed[col])
                
        if self.scaling:
            X_transformed = self.scaler.transform(X_transformed)
            
        return X_transformed
    
    def fit_transform(self, X):
        """Fit the preprocessor and transform the data.
        
        Parameters
        ----------
        X : array-like
            Data to fit and transform
            
        Returns
        -------
        array-like
            Transformed data
        """
        return self.fit(X).transform(X)