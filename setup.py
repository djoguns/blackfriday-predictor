import os
from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as fp:
    install_requires = fp.read()

setup(
    name='blackfridaylib',
    version='0.1',
    install_requires=install_requires,
    packages=find_packages(),
    include_package_data=True,
    author='Mohamed Leila',
    scripts=[os.path.join('bin', 'train'),
            os.path.join('bin', 'predict')],
    description='Predicting Black Friday purchases'
)
