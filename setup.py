import sys

try:
    from skbuild import setup
    import nanobind
    import torch


except ImportError:
    print("The preferred way to invoke 'setup.py' is via pip, as in 'pip "
          "install .'. If you wish to run the setup script directly, you must "
          "first install the build dependencies listed in pyproject.toml!",
          file=sys.stderr)
    raise

setup(
    name="fast_altmin",
    version="0.0.1",
    author="Tom Maguire",
    author_email="thomas.d.maguire@btinternet.com",
    description="cpp implementation of alternating minimisation",
    url="https://github.com/DiamonDinoia/altmin",
    license="BSD",
    packages=['fast_altmin'],
    package_dir={'': 'src'},
    cmake_install_dir="src/fast_altmin",
    include_package_data=True,
    python_requires=">=3.8"
)
