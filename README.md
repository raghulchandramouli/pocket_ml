# pocket_ml

A lightweight and user-friendly machine learning library designed to simplify ML workflows.

## Features

- Simple and intuitive API for common ML tasks
- Automated data preprocessing
- Easy model training and evaluation
- Built-in visualization tools
- Comprehensive documentation and examples

## Installation

```bash
pip install pocket_ml
```

## Quick Start

```python
from pocket_ml.model import Classifier
from pocket_ml.preprocessing import DataPreprocessor

# Prepare your data
preprocessor = DataPreprocessor()
X_processed = preprocessor.fit_transform(X)

# Train a model
model = Classifier('random_forest')
model.fit(X_processed, y)

# Make predictions
predictions = model.predict(new_data)
```

## Documentation

For detailed documentation and examples, visit our [documentation page](https://pypi.org/project/pocket-ml/0.1.1/).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
