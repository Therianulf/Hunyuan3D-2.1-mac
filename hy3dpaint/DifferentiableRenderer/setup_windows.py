"""Windows build for mesh_inpaint_processor (pybind11 extension).

compile_mesh_painter.sh only covers macOS/Linux -- it shells out to
`python3-config --extension-suffix` and passes -fPIC/-shared, none of which
exist under MSVC. This is the Windows equivalent.

Build in place (from this directory, with the venv active):
    ..\\..\\.venv\\Scripts\\python.exe setup_windows.py build_ext --inplace

Produces mesh_inpaint_processor.cp311-win_amd64.pyd next to MeshRender.py,
which is exactly where `from .mesh_inpaint_processor import meshVerticeInpaint`
(MeshRender.py:37) looks for it.
"""

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "mesh_inpaint_processor",
        ["mesh_inpaint_processor.cpp"],
        cxx_std=17,
        # /utf-8: the source carries UTF-8 Chinese comments. Without this MSVC
        # decodes them as the system ANSI codepage and emits C4819, which turns
        # fatal under /WX-style setups.
        # /O2: match the -O3 the POSIX script asks for.
        extra_compile_args=["/utf-8", "/O2"],
    ),
]

setup(
    name="mesh_inpaint_processor",
    version="0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
