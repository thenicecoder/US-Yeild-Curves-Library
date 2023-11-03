from setuptools import setup, find_packages

setup(
    name='yield_curve_lib',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scikit-learn',
        'ipywidgets'
    ],
)
