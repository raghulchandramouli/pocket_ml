from setuptools import setup, find_packages

setup(
    name="pocket_ml",
    version="0.1.0",
    packages=find_packages(),
    package_data={
        '': ['LICENSE', 'README.md']
    },
    install_requires=[
        'numpy>=1.19.0',
        'pandas>=1.2.0',
        'scikit-learn>=0.24.0',
        'matplotlib>=3.3.0',
        'seaborn>=0.11.0'
    ],
    author="Raghul",
    author_email="your.email@example.com",
    description="A simplified machine learning library for easy ML workflows",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/raghulchandramouli/pocket_ml",
    project_urls={
        "Documentation": "https://pocket_ml.readthedocs.io",
        "Bug Reports": "https://github.com/raghulchandramouli/pocket_ml/issues",
        "Source Code": "https://github.com/raghulchandramouli/pocket_ml"
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    keywords="machine learning, data science, classification, visualization",
    python_requires='>=3.7',
    license="MIT",
)