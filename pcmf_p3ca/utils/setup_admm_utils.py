# utils/setup_admm_utils.py
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        name="pcmf_p3ca.utils.admm_utils",                # import path
        sources=["pcmf_p3ca/utils/admm_utils.pyx"],       # .pyx path
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=["-Wno-unreachable-code", "-Wno-deprecated-declarations"],
    ),
]

setup(
    name="admm_tools",
    ext_modules=cythonize(ext_modules, language_level="3"),
)
