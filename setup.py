from setuptools import setup, find_packages
from glob import glob

setup(
   name='miricoord',
   version='1.0.0',
   description='MIRI coordinate transforms',
   author='David R. Law',
   author_email='dlaw@stsci.edu',
   packages=find_packages(),
   data_files=[
       ('data/crds',glob('data/crds/*')),
       ('data/dithers',glob('data/dithers/*')),
       ('data/fits/cdp6',glob('data/fits/cdp6/*')),
       ('data/fits/cdp7',glob('data/fits/cdp7/*')),
       ('data/fits/cdp7beta3',glob('data/fits/cdp7beta3/*')),
       ('data/fits/cdp8b',glob('data/fits/cdp8b/*')),
   ],
   include_package_data=True,
   install_requires=['jwst', 'pysiaf', 'jupyter', 'matplotlib'],
)
