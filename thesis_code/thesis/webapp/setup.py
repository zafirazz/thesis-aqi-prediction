from setuptools import setup, find_packages

setup(
    name="thesis",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "pandas",
        "streamlit-option-menu",
        "numpy"
    ]
)