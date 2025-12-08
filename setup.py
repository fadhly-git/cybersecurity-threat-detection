"""Setup configuration for cybersecurity threat detection package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="cybersecurity-threat-detection",
    version="1.0.0",
    author="Cybersecurity Research Team",
    description="ML/DL implementation for cybersecurity threat detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fadhly-git/cybersecurity-threat-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-ml=scripts.train_ml_models:main",
            "train-dl=scripts.train_dl_models:main",
            "evaluate=scripts.evaluate_models:main",
            "run-pipeline=scripts.run_pipeline:main",
        ],
    },
)
