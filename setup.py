from setuptools import setup, find_packages

setup(
   name='miricoord',
   version='1.0.0',
   description='MIRI coordinate transforms',
   author='David R. Law',
   author_email='dlaw@stsci.edu',
   packages=find_packages(),
   install_requires=['jwst', 'pysiaf', 'jupyter', 'matplotlib'],
)
