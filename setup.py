from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "cfits", ["tests/benchmark/snippets/cfits.pyx"],
        language="c++",
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
        )
]

setup(
    ext_modules=cythonize(extensions)
)