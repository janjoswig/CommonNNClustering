import setuptools
from Cython.Build import cythonize

with open("README.md", "r") as readme:
    desc = readme.read()

extensions = [
    setuptools.Extension(
        "cnnclustering._cfits", ["cnnclustering/_cfits.pyx"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        )
]

setuptools.setup(
    name='cnnclustering',
    version='0.3',
    keywords=["Density-based-clustering"],
    scripts=["cnnclustering/cnn.py", "cnnclustering/cmsm.py", "cnnclustering/_plots.py"],
    author="Jan-Oliver Joswig",
    author_email="jan.joswig@fu-berlin.de",
    description="A Python package for common-nearest neighbour (CNN) clustering",
    long_description=desc,
    long_description_content_type="text/markdown",
    url="https://github.com/janjoswig/CNN",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        ],
    ext_modules=cythonize(extensions),
    # cmdclass={'build_ext': Cython.Build.new_build_ext}
    )
