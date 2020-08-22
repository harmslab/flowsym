#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', 'pytest~=6.0.1', 'numpy~=1.15.4','pandas~=0.23.4', 'hdbscan~=0.8.26', 'matplotlib~=3.0.2',
                'seaborn~=0.9.0', 'fcsy~=0.4.0', 'unidip~=0.1.1', 'scipy~=1.1.0', 'scikit-learn~=0.20.1']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Luis Perez Morales, Michael M. Shavlik",
    author_email='lperezmo@uoregon.edu, mshavlik@uoregon.edu',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="A Python API for simulating flow cytometry data",
    entry_points={
        'console_scripts': [
            'flowsym=flowsym.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='flowsym',
    name='flowsym',
    packages=find_packages(include=['flowsym', 'flowsym.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/mshavlik; lperezmo/flowsym',
    version='0.1.0',
    zip_safe=False,
)
