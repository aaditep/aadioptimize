from setuptools import setup

setup(
    name="aadioptimize",
    version='0.0.1',
    description='Perform optimization for function',
    url='https://github.com/aaditep/aadioptimize',
    author='Aadi Tepper',
    author_email='aaditep@gmail.com',
    licence='MIT',
    packages=['aadioptimize'],
    install_requires=[
        'numpy.matlib',
        'numpy',
    ],
    python_requires='>=3.7'
)
