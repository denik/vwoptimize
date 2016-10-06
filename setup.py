from distutils.core import setup
import re

version = re.search(r"__version__\s*=\s*'(.*)'", open('vwoptimize.py').read(500), re.M).group(1)
assert version

setup(name='vwoptimize',
      version=version,
      description='Cross-validation and hyper-parameter search for Vowpal Wabbit',
      author='Denis Bilenko',
      author_email='denis.bilenko@gmail.com',
      url='https://github.com/denik/vwoptimize',
      py_modules=['vwoptimize'],
      scripts=['vwoptimize.py']
      )
