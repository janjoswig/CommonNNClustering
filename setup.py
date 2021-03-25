import os
import sysconfig
from setuptools import Extension, find_packages, setup
from typing import List, Optional, Tuple

from Cython.Build import cythonize
import numpy as np


PYTHON_REQUIRES = ">=3.6"
TRACE_CYTHON = bool(int(os.getenv("TRACE_CYTHON", 0)))

cython_macros: List[Tuple[str, Optional[str]]] = [
    ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")
]

if TRACE_CYTHON:
    cython_macros.append(("CYTHON_TRACE", None))
    cython_macros.append(("CYTHON_TRACE_NOGIL", None))

extra_compile_args = set(sysconfig.get_config_var('CFLAGS').split())
extra_compile_args.discard('-Wstrict-prototypes')
extra_compile_args.add("-fno-var-tracking-assignments")
extra_compile_args.add("-std=c++14")

extensions = [
    Extension(
        "*",
        ["src/cnnclustering/*.pyx"],
        define_macros=cython_macros,
        language="c++",
        include_dirs=[np.get_include()],
        extra_compile_args=list(extra_compile_args),
    )
]

compiler_directives = {
    "language_level": 3,
    "embedsignature": True,
    "boundscheck": False,
    "wraparound": False,
    "cdivision": True,
    "nonecheck": False,
    "linetrace": True,
}

extensions = cythonize(extensions, compiler_directives=compiler_directives)

with open("README.md", "r") as readme:
    desc = readme.read()

sdesc = "A Python package for common-nearest-neighbours clustering"

requirements_map = {
    "mandatory": "",
    "optional": "-optional",
    "dev": "-dev",
    "docs": "-docs",
    "test": "-test",
}

requirements = {}
for category, fname in requirements_map.items():
    with open(f"requirements{fname}.txt") as fp:
        requirements[category] = fp.read().strip().split("\n")

setup(
    name="cnnclustering",
    version="0.3.11",
    keywords=["Density-based clustering"],
    author="Jan-Oliver Joswig",
    author_email="jan.joswig@fu-berlin.de",
    description=sdesc,
    long_description=desc,
    long_description_content_type="text/markdown",
    url="https://github.com/janjoswig/CommonNNClustering",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"cnnclustering": ["*.pxd"]},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    ext_modules=extensions,
    install_requires=requirements["mandatory"],
    extras_require={
        "optional": requirements["optional"],
        "dev": requirements["dev"],
        "docs": requirements["docs"],
        "test": requirements["test"],
    },
    zip_safe=False,
)
