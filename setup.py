from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os

from setuptools import setup, find_packages

install_requires = [
    "torch>=1.10.0",
    "numpy>=1.17.2",
    "scipy>=1.6.0",
    "pandas>=1.0.5",
    "tqdm>=4.48.2",
    "scikit_learn>=0.23.2",
    "pyyaml>=5.1.0",
    "tensorboard>=2.5.0",
    "requests>=2.27.1",
    "pytz>=2022.1",
    "matplotlib>=3.5.1",
    "deepdiff>=6.3.1",
    "networkx>=2.8"
]

setup_requires = []

extras_require = {}

classifiers = ["License :: OSI Approved :: MIT License"]

long_description = (
    "EduStudio is a Unified and Templatized Framework "
    "for Student Assessment Models including "
    "Cognitive Diagnosis(CD) and Knowledge Tracing(KT) based on Pytorch."
)

# Readthedocs requires Sphinx extensions to be specified as part of
# install_requires in order to build properly.
on_rtd = os.environ.get("READTHEDOCS", None) == "True"
if on_rtd:
    install_requires.extend(setup_requires)

setup(
    name="edustudio",
    version="v1.1.3", 
    description="a Unified and Templatized Framework for Student Assessment Models",
    long_description=long_description,
    python_requires='>=3.8',
    long_description_content_type="text/markdown",
    url="https://github.com/HFUT-LEC/EduStudio",
    author="HFUT-LEC",
    author_email="lmcRS.hfut@gmail.com",
    packages=[package for package in find_packages() if package.startswith("edustudio")],
    include_package_data=True,
    install_requires=install_requires,
    setup_requires=setup_requires,
    extras_require=extras_require,
    zip_safe=False,
    classifiers=classifiers,
    entry_points={
        "console_scripts": [
            "edustudio = edustudio.quickstart.atom_cmds:entrypoint",
        ],
    }
)
