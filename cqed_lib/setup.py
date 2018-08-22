from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

EXT_MODULES = [Extension("cqed_tools.simulation.gsl", ["cqed_tools/simulation/gsl.pyx"],libraries=["gsl","gslcblas"],include_dirs=["/homes/pbrookes/gsl/include"],library_dirs=["/homes/pbrookes/gsl/lib"])]

setup(name='cqed_tools',
      version='1.0.0',
      description='Tools for simulating the dynamics of a drive cavity coupled to a transmon qubit.',
      packages=find_packages(),
      ext_modules = cythonize(EXT_MODULES))

