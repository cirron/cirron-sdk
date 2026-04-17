from setuptools import setup, find_packages

setup(
    name="cirron",
    version="0.0.1",
    description="SDK for ML engineers to integrate with Cirron",
    author="Devin Lynch",
    author_email="devin@cirron.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.0.0",
        "pyyaml>=5.1",
        "pydantic>=2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.0.0",
        ],
        "pytorch": ["torch>=1.7.0"],
        "tensorflow": ["tensorflow>=2.4.0"],
        "sklearn": ["scikit-learn>=0.24.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)