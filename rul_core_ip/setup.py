from setuptools import find_packages, setup

setup(
    name="rul_core_ip",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "hydra-core>=1.3.2",
        "mlflow>=2.22.0",
        "pandas>=2.2.3",
        "scikit-learn>=1.6.1",
        "boto3>=1.37.3",
        "numpy>=2.2.0",
        "omegaconf>=2.3.0",
        "joblib>=1.5.0",
    ],
    author="RUL Team",
    description="Proprietary RUL prediction core IP package",
)
