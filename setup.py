from setuptools import setup, find_packages


with open("README.md", "r") as f:
    description = f.read()
    
setup(
    name="pocket_ml",  # You might need to change this if the name is taken
    version="0.1.2",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19.0',
        'pandas>=1.2.0',
        'scikit-learn>=0.24.0',
        'matplotlib>=3.3.0',
        'seaborn>=0.11.0'
    ],
    
    long_description=description,
    long_description_content_type="text/markdown",
)