from setuptools import setup, find_packages

setup(
    name="unisite",
    version="1.0",
    packages=find_packages(exclude=["config",]),
)
