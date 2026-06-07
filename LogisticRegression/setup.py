from setuptools import setup, find_packages

setup(
    name="mlops-logistic-regression",
    version="1.0.0",
    description="End-to-end MLOps pipeline with Logistic Regression",
    author="Ramakrushna Mohapatra",
    author_email="itsramakrushna@gmail.com",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "mlflow>=2.9.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.4.0",
        "pyyaml>=6.0.1",
        "loguru>=0.7.0",
        "joblib>=1.3.0",
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov", "ruff", "black", "httpx"],
        "monitoring": ["evidently>=0.4.0", "plotly>=5.17.0"],
    },
)
