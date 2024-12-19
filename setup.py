from setuptools import setup, find_packages

setup(
    name="sheetz-recommender",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "google-cloud-storage>=2.14.0",
        "google-cloud-bigquery>=3.17.1",
        "torchrec>=0.6.0",
        "torch>=2.0.0",
        "pyyaml>=6.0.1",
        "pandas>=2.0.0",
        "numpy>=1.24.0"
    ],
    python_requires=">=3.8",
)
