#!/usr/bin/env python

from setuptools import find_packages, setup

packages = ["superqulan." + p for p in find_packages("superqulan")]

setup(
    name="SuperQuLAN",
    version="0.1",
    description="SuperQuLAN superconducting Quantum Link Modelization Library",
    url="https://www.superqulan.eu/",
    packages=["superqulan"] + packages,
)

