from distutils.core import setup

setup(name='vwoptimize',
      version='0.1',
      description='Cross-validation and hyper-parameter search for Vowpal Wabbit',
      author='Denis Bilenko',
      author_email='denis.bilenko@gmail.com',
      url='https://github.com/denik/vwoptimize',
      py_modules=['vwoptimize'],
      scripts=['vwoptimize.py']
      )
