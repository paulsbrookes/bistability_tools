from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

EXT_MODULES = [Extension("cqed_tools.cy.gsl", ["cqed_tools/cy/gsl.pyx"],libraries=["gsl","gslcblas"])]

setup(name='cqed_tools',
      version='1.0.0',
      description='Tools for simulating the dynamics of a drive cavity coupled to a transmon qubit.',
      packages=find_packages(),
      ext_modules = cythonize(EXT_MODULES))

