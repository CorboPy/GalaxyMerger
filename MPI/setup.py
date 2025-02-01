# Compile MPI/leapfrog.pyx

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Define the Cython extension module with OpenMP support
extensions = [
    Extension(
        name="leapfrog",  # This is the name of the output module
        sources=["leapfrog.pyx"],  # The Cython file to compile
        include_dirs=[numpy.get_include()],  # Include NumPy headers for Cython
#        libraries=["m"],  # Link against the math library (for sqrt)
        extra_compile_args=["-fopenmp"],  # Enable OpenMP for the compiler
        extra_link_args=["-fopenmp"],  # Enable OpenMP for the linker
    )
]

# Setup the module
setup(
    name="Leapfrog Integration Module",
    ext_modules=cythonize(
        extensions,
        compiler_directives={"boundscheck": False, "wraparound": False}, 
    ),
    include_dirs=[numpy.get_include()],  # Ensure NumPy headers are included
)