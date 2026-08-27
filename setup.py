from setuptools import setup, find_packages
from pathlib import Path
import os
import sys

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME


def resolve_requirements(file):
    requirements = []
    with open(file) as f:
        req = f.read().splitlines()
        for r in req:
            if r.startswith("-r"):
                requirements += resolve_requirements(
                    os.path.join(os.path.dirname(file), r.split(" ")[1]))
            else:
                requirements.append(r)
    return requirements


def read_file(file):
    with open(file) as f:
        content = f.read()
    return content


def clean():
    """Custom clean command to tidy up the project root."""
    os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz')


def get_extensions():
    """
    Adapted from https://github.com/pytorch/vision/blob/master/setup.py
    and https://github.com/facebookresearch/detectron2/blob/master/setup.py
    """
    print("Build csrc")
    print("Building with {}".format(sys.version_info))

    this_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    extensions_dir = this_dir/'nndet'/'csrc'

    main_file = list(extensions_dir.glob('*.cpp'))
    source_cpu = []  # list((extensions_dir/'cpu').glob('*.cpp')) temporary until I added header files ...
    source_cuda = list((extensions_dir/'cuda').glob('*.cu'))
    print("main_file {}".format(main_file))
    print("source_cpu {}".format(source_cpu))
    print("source_cuda {}".format(source_cuda))

    sources = main_file + source_cpu
    extension = CppExtension

    define_macros = []
    extra_compile_args = {"cxx": []}

    if (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv('FORCE_CUDA', '0') == '1':
        print("Adding CUDA csrc to build")
        print("CUDA ARCH {}".format(os.getenv("TORCH_CUDA_ARCH_LIST")))
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [('WITH_CUDA', None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
        
        # It's better if pytorch can do this by default ..
        CC = os.environ.get("CC", None)
        if CC is not None:
            extra_compile_args["nvcc"].append("-ccbin={}".format(CC))

    # Set RPATH to find PyTorch libraries at runtime
    # This ensures the extension can find libc10.so, libtorch.so, etc.
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
    extra_link_args = []
    if sys.platform.startswith('linux'):
        extra_link_args = [
            f'-Wl,-rpath,{torch_lib_path}',           # Absolute path to build-time torch
            '-Wl,-rpath,$ORIGIN/../torch/lib',        # Relative path (if torch is sibling in venv)
            '-Wl,-rpath,$ORIGIN/../../torch/lib',     # Alternative relative path
        ]

    # Convert Path objects to relative strings from setup.py directory
    sources = [str(s.relative_to(this_dir)) for s in sources]
    include_dirs = [str(extensions_dir.relative_to(this_dir))]
    
    ext_modules = [
        extension(
            'nndet._C',
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]
    
    return ext_modules

# Note: Dependencies are defined in pyproject.toml [project] section
# setup.py is only used for building C++ extensions

setup(
    name='nndet',
    version="0.3.0",  # Synced with pyproject.toml
    packages=find_packages(),
    python_requires=">=3.10,<3.14",  # Synced with pyproject.toml
    author="Division of Medical Image Computing, German Cancer Research Center",
    maintainer_email='m.baumgartner@dkfz-heidelberg.de',
    ext_modules=get_extensions(),
    cmdclass={
        'build_ext': BuildExtension,
        'clean': clean,
    },
    entry_points={
        'console_scripts': [
            'nndet_example = nndet.scripts.generate_example:main',

            'nndet_prep = nndet.scripts.preprocess:main',
            'nndet_cls2fg = nndet.scripts.convert_cls2fg:main',
            'nndet_seg2det = nndet.scripts.convert_seg2det:main',

            'nndet_train = nndet.scripts.train:train',
            'nndet_sweep = nndet.scripts.train:sweep',

            'nndet_eval = nndet.scripts.train:evaluate',
            'nndet_predict = nndet.scripts.predict:main',
            'nndet_consolidate = nndet.scripts.consolidate:main',

            'nndet_boxes2nii = nndet.scripts.utils:boxes2nii',
            'nndet_seg2nii = nndet.scripts.utils:seg2nii',
            'nndet_unpack = nndet.scripts.utils:unpack',
            'nndet_env = nndet.scripts.utils:env',
            'nndet_searchpath = nndet.scripts.utils:hydra_searchpath'
        ]
    },
)
