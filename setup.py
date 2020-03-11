from setuptools import setup

setup(
   name='miricoord',
   version='1.0.0',
   description='MIRI coordinate transforms',
   author='David R. Law',
   author_email='dlaw@stsci.edu',
   packages=['miricoord'],
   install_requires=['jwst', 'pysiaf', 'jupyter', 'matplotlib'],
)
