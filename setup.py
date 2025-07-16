#!/usr/bin/env python3
"""
Setup script for Python Machine Learning and Computer Vision Project
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "Python Machine Learning and Computer Vision Project"

# Read requirements from requirements.txt
def read_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh.readlines() if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return []

setup(
    name="ml-cv-project",
    version="0.1.0",
    author="ML/CV Developer",
    author_email="developer@example.com",
    description="Python机器学习和计算机视觉项目，集成PyTorch、OpenCV、PyQt5等核心库",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/username/ml-cv-project",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
            "jupyterlab>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ml-cv-train=src.models.train:main",
            "ml-cv-gui=gui.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    project_urls={
        "Bug Reports": "https://github.com/username/ml-cv-project/issues",
        "Source": "https://github.com/username/ml-cv-project",
        "Documentation": "https://ml-cv-project.readthedocs.io/",
    },
)