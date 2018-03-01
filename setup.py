from setuptools import setup, find_packages
import re

version = re.search(r"__version__\s*=\s*'(.*)'", open('vwoptimize.py').read(500), re.M).group(1)
assert version

setup(name='vwoptimize',
      version=version,
      description='Hyper-parameter search, text preprocessing and reporting for Vowpal Wabbit',
      author='Denis Bilenko',
      author_email='denis.bilenko@gmail.com',
      url='https://github.com/denik/vwoptimize',
      packages=find_packages('.'),
      scripts=['vwoptimize.py']
      )
