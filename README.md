# pocket_ml

[![PyPI version](https://badge.fury.io/py/pocket-ml.svg)](https://badge.fury.io/py/pocket-ml)

A lightweight and user-friendly machine learning library designed to simplify ML workflows.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Package Structure](#package-structure)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Features

pocket_ml offers a range of features to streamline your machine learning projects:

-   **Simple and Intuitive API:** Provides easy-to-use classes like `Classifier`, `DataPreprocessor`, and `Visualizer` for common ML tasks, reducing boilerplate code.
-   **Automated Data Preprocessing:** Includes the `DataPreprocessor` class to handle essential preprocessing steps like scaling, encoding categorical features, and handling missing values (functionality may vary based on implementation details).
-   **Easy Model Training and Evaluation:** Train various classification and regression models with a consistent `.fit()` and `.predict()` interface. Evaluate model performance using standard metrics.
-   **Built-in Visualization Tools:** The `Visualizer` class helps in understanding data and model results through plots like confusion matrices, feature importance plots, etc. (specific plots depend on implementation).
-   **Comprehensive Documentation and Examples:** Access detailed guides and usage examples to get started quickly.

## Installation

```bash
pip install pocket_ml
```

## Quick Start

Here's a basic example of how to use pocket_ml:

```python
from pocket_ml import Classifier, DataPreprocessor, Visualizer

# Assume X is your feature matrix (e.g., pandas DataFrame or NumPy array)
# Assume y is your target vector (e.g., pandas Series or NumPy array)
# Assume new_data is the data you want to make predictions on

# 1. Prepare your data
preprocessor = DataPreprocessor() # Initialize the preprocessor
X_processed = preprocessor.fit_transform(X) # Apply preprocessing
# Preprocess the new_data similarly (using transform, not fit_transform)
# new_data_processed = preprocessor.transform(new_data)

# 2. Train a model
# Choose a model type (e.g., 'random_forest', 'logistic_regression')
model = Classifier(model_type='random_forest')
model.fit(X_processed, y) # Train the model

# 3. Make predictions
# Ensure new_data is preprocessed using the *same* preprocessor instance
# predictions = model.predict(new_data_processed)

# 4. Visualize results (Example for classification)
# Assuming you have true labels (y_test) and predictions for a test set
# visualizer = Visualizer()
# visualizer.plot_confusion_matrix(y_test, predictions)
```

## Package Structure

The library is organized as follows:

```
pocket_ml/
  ├── __init__.py         # Makes pocket_ml a package
  ├── algorithms/         # ML algorithms implementation
  │   ├── __init__.py
  │   ├── classification/ # Classification algorithms
  │   └── regression/     # Regression algorithms
  ├── preprocessing/      # Data preprocessing utilities
  │   ├── __init__.py
  │   └── data_preprocessor.py
  └── visualization/      # Data visualization tools
      ├── __init__.py
      └── visualizer.py
```

## Documentation

For detailed documentation and examples, visit our [documentation page](https://pypi.org/project/pocket-ml/). (Note: The provided link points to version 0.1.2, ensure documentation matches the installed version).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Refer to the project's contribution guidelines if available.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details (if included in the repository).
