
import os

from setuptools import find_namespace_packages, setup

REQUIRES = [
    "requests>=2.20.0",
    "globus-sdk>=3.12.0",
]

TEST_REQUIRES = [
    "flake8==3.8.0",
    "pytest"
]
DEV_REQUIRES = TEST_REQUIRES + [
    "pre-commit",
]

# This will be set in GitHub as part of the release
version = os.getenv("garden_version")
if version is None:
    version = "0.1a1"
else:
    version = version.split("/")[-1]

# Use the readme as the long description.
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="garden-ai",
    version=version,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_namespace_packages(include=["garden"]),
    description="Garden: tools to simplify access to scientific AI advances.",
    install_requires=REQUIRES,
    extras_require={
        "dev": DEV_REQUIRES,
        "test": TEST_REQUIRES
    },
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
    ],
    keywords=["garden", "ML", "AI"],
    author="Garden team",
    license="Apache License, Version 2.0",
    url="https://github.com/Garden-AI/garden",
)
