
import os
import re

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

def parse_version():
    # single source of truth for package version
    version_string = ""
    version_pattern = re.compile(r'__version__ = "([^"]*)"')
    with open(os.path.join("garden", "version.py")) as f:
        for line in f:
            match = version_pattern.match(line)
            if match:
                version_string = match.group(1)
                break
    if not version_string:
        raise RuntimeError("Failed to parse version information")
    return version_string

version = os.getenv("garden_version")
if version is None:
    version = "0.1a1"
else:
    version = version.split("/")[-1]

setup(
    name="garden",
    version=version,
    packages=find_namespace_packages(include=["garden"]),
    description="Garden: a collection of tools to simplify access to scientific AI advances.",
    install_requires=REQUIRES,
    extras_require={
        "dev": DEV_REQUIRES,
        "test": TEST_REQUIRES
    },
    python_requires=">=3.8",
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