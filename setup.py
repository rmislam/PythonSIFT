"""Setup for pysift."""

from setuptools import find_packages, setup

with open("README.md") as f:
    readme = f.read()

# Runtime requirements.
inst_reqs = [
    "numpy",
    "opencv-python",
]

extra_reqs = {
    "test": ["pytest", "pytest-benchmark", "pytest-cov"],
    "dev": ["pytest", "pytest-benchmark", "pytest-cov"],
    "examples": ["matplotlib"],
    "docs": ["mkdocs"],
}

setup(
    name="pysift",
    version="0.0.1",
    python_requires=">=3.5",
    description="Python implementation of the SIFT algorithm.",
    long_description=readme,
    long_description_content_type="text/markdown",
    classifiers=[
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering",
    ],
    keywords="Image processing",
    author="Russ Islam",
    author_email="russmislam@gmail.com",
    url="https://github.com/rmislam/PythonSIFT",
    license="MIT",
    packages=find_packages(exclude=["ez_setup", "examples", "tests"]),
    include_package_data=True,
    zip_safe=True,
    install_requires=inst_reqs,
    extras_require=extra_reqs,
)
