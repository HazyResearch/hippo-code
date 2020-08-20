from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

ext_modules = []
extension = CppExtension('hippo', ['hippo.cpp', 'hippolegs.cpp', 'hippolegt.cpp'], extra_compile_args=['-march=native'])
ext_modules.append(extension)

setup(
    name='hippo',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension})
