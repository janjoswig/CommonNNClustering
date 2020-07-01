# import os
from setuptools import setup, find_packages, Extension

from Cython.Build import cythonize
from Cython.Distutils import build_ext

# try:
#     from Cython.Build import cythonize
#     from Cython.Distutils import build_ext
# except ImportError:
#     cythonize = None

import numpy as np


# def no_cythonize(extensions, **_ignore):
#     for extension in extensions:
#         sources = []
#         for sfile in extension.sources:
#             path, ext = os.path.splitext(sfile)
#             if ext in (".pyx", ".py"):
#                 if extension.language == "c++":
#                     ext = ".cpp"
#                 else:
#                     ext = ".c"
#                 sfile = path + ext
#             sources.append(sfile)
#         extension.sources[:] = sources
#    return extensions


extensions = [
    Extension(
        "cnnclustering._cfits", ["cnnclustering/_cfits.pyx"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        language="c++",
        include_dirs=[np.get_include()]
        )
]

# NOCYTHONIZE = bool(int(os.getenv("NOCYTHONIZE", 1))) or cythonize is None

# if NOCYTHONIZE:
#     extensions = no_cythonize(extensions)
# else:
compiler_directives = {
    "language_level": 3,
    "embedsignature": True,
    "cython: boundscheck": False,
    "cython: wraparound": False,
    "cython: cdivision": True,
    "cython: nonecheck": False
    }
extensions = cythonize(extensions, compiler_directives=compiler_directives)

with open("README.md", "r") as readme:
    desc = readme.read()

sdesc = "A Python package for common-nearest neighbour (CNN) clustering"

requirements_map = {"mandatory": "",
                    "optional": "-optional",
                    "dev": "-dev",
                    "docs": "-docs",
                    "tests": "-tests"}

requirements = {}
for category, fname in requirements_map.items():
    with open(f"requirements{fname}.txt") as fp:
        requirements[category] = fp.read().strip().split("\n")

setup(
    name='cnnclustering',
    version='0.3.8',
    keywords=["Density-based-clustering"],
    scripts=["cnnclustering/cnn.py",
             "cnnclustering/cmsm.py",
             "cnnclustering/_plots.py"],
    author="Jan-Oliver Joswig",
    author_email="jan.joswig@fu-berlin.de",
    description=sdesc,
    long_description=desc,
    long_description_content_type="text/markdown",
    url="https://github.com/janjoswig/CNN",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        ],
    ext_modules=extensions,
    install_requires=requirements["mandatory"],
    extras_require={
        "optional": requirements["optional"],
        "dev": requirements["dev"],
        "docs": requirements["docs"],
        "tests": requirements["tests"],
        },
    cmdclass=dict(build_ext=build_ext)
    )
