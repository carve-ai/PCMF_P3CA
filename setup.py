import io
import os
from setuptools import setup, find_packages

short_description = "PCMF and Related Algorithms: PCMF, LL-PCMF, and P3CA are designed to implement cluster-aware embedding on single-view and two-view datasets."
long_description = "Pathwise Clustered Matrix Factorization (PCMF), Locally Linear PCMF (LL-PCMF), and Pathwise Clustered CCA (P3CA), as described in the AISTATS 2024 paper (Buch, Liston, & Grosenick. AISTATS. 2024)."
long_description_content_type = "text/markdown"

here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = short_description

setup(
    name="pcmf_p3ca",
    version="1.0.0",
    description=short_description,
    long_description=long_description,
    long_description_content_type=long_description_content_type,
    python_requires=">=3.8",
    author="Amanda M. Buch",
    author_email="amanda.m.buch@gmail.com",
    url="https://github.com/carve-ai/pcmf_p3ca",
    include_package_data=True,
    install_requires=[
        "cvxpy>=1.5.3",
        "jupyter",
        "matplotlib>=3.5.2",
        "numba>=0.56.4",
        "numpy>=1.23.4",
        "palmerpenguins>=0.1.4",
        "pandas>=1.3.4",
        "scikit-learn>=1.2.0",
        "scikit-sparse>=0.4.14",
        "scipy>=1.9.3",
        "seaborn>=0.13.2",
        "tqdm>=4.67.1",
    ],
    extras_require={
        "mosek": ["mosek"],          
        "cython_ext": ["cython>=3.0.11"]
    },
    license="Proprietary License (see license file)",
    project_urls={
    "Copyright": "Copyright (c) 2022–present Amanda M. Buch, Conor Liston, and Logan Grosenick.",
    "Source": "https://github.com/carve-ai/pcmf_p3ca",
    },
    classifiers=[
        "License :: Other/Proprietary License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Intended Audience :: Science/Research (Non-commercial)",
        "Programming Language :: Python :: 3",
    ],
)