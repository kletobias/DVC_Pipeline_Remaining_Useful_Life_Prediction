# setup.py
from setuptools import find_packages, setup

setup(
    name="portfolio_predictive_maintenance_rul",
    version="0.1.1",
    packages=find_packages(include=["dependencies*", "scripts*"]),
)
