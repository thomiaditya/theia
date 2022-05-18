# Copyright (C) 2022 by Thomi Aditya.  All rights reserved.

from setuptools import setup, find_packages

# Get the long description from the README file
with open('README.md', encoding='utf-8') as f:
  long_description = f.read()

with open('requirements.txt') as f:
  requirements = f.read().splitlines()


setup(
  name='theia',
  version='0.0.1',
  description='Sentiment analysis using the Keras library',
  long_description=long_description,
  long_description_content_type='text/markdown',
  url='https://github.com/thomiaditya/theia',
  author='Thomi Aditya Alhakiim',
  license='MIT',
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
  ],

  keywords='sentiment analysis',
  packages=find_packages(exclude=['docs', 'tests']),
  install_requires=requirements,
  python_requires='>=3.7',
)