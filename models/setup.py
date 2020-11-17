from setuptools import setup
import setuptools

setup(
    name='UMNN',
    version='1.62',
    author="Antoine Wehenkel",
    author_email="antoine.wehenkel@gmail.com",
    description="Implementation of unconstrained monotonic neural networks and their application for autoregressive normalizing flows.",
    long_description='',
    long_description_content_type="text/markdown",
    url="https://github.com/awehenkel/umnn",
    packages=setuptools.find_packages(),
    classifiers=[
     "Programming Language :: Python :: 3",
     "License :: OSI Approved :: MIT License",
     "Operating System :: OS Independent",
     ],
)

