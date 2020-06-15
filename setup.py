import os
from setuptools import setup, find_packages, Extension

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None


def no_cythonize(extensions, **_ignore):
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in (".pyx", ".py"):
                if extension.language == "c++":
                    ext = ".cpp"
                else:
                    ext = ".c"
                sfile = path + ext
            sources.append(sfile)
        extension.sources[:] = sources
    return extensions


extensions = [
    Extension(
        "cnnclustering._cfits", ["cnnclustering/_cfits.pyx"],
        # include_path = [numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        language="c++"
        )
]

CYTHONIZE = bool(int(os.getenv("CYTHONIZE", 0))) and cythonize is not None

if CYTHONIZE:
    compiler_directives = {"language_level": 3,
                           "embedsignature": True,
                           "cython: boundscheck": False,
                           "cython: wraparound": False,
                           "cython: cdivision": True,
                           "cython: nonecheck": False}
    extensions = cythonize(extensions, compiler_directives=compiler_directives)
else:
    extensions = no_cythonize(extensions)


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
    version='0.3.2',
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
    )
