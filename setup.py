from distutils.command.build import build as _build
import setuptools
from setuptools import setup, find_packages

setup(name='gan',
        version='0.1',
        packages=find_packages(),
        description='example to run keras on gcloud ml-engine',
        author='TaehoLee',
        author_email='smartdolphin07@gmail.com',
        license='MIT',
        install_requires=[
			'keras',
			'h5py',
			'matplotlib',
			'numpy',
			'six',
			'pyyaml'
		],
        zip_safe=False)
